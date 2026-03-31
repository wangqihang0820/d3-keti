# feature_extractors/recon_features.py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 添加这行

from models.models import Model  # 复用原来的 2D/3D 编码器
from utils.utils import KNNGaussianBlur
from feature_extractors.change_ex import CombinedExchange   # ★ 新增：用你的 CSS 模块
from models.cross_attention import BiDirectionalCrossAttention  # ← 新增
from models.lapwavegate import LapWaveGate
from utils.ot_fusion import fuse_by_task_ot,robust_zscore
from feature_extractors.ReconNet2D import ReconNet2D
from feature_extractors.ReconNet3D import ReconNet3D
from feature_extractors.shared_basis import SharedBasis  # <--- 新增
from feature_extractors.masking import LatentRandomMasking
from feature_extractors.spectral_branch import *
from feature_extractors.spatial_branch import SpatialComplementBranch
from feature_extractors.fusion_spatial_spectral import DualStreamFusion
from feature_extractors.ot_module import UncertaintyAwareOT



# ------------------------------------------------
# 主干网络：2D / ps 编码 + CSS + 3D 编码 + Cross-Attn + 重建
# ------------------------------------------------
class FusionReconNet(nn.Module):
    """
    主干网络：
      - 2D RGB & ps: 先各自通过 ViT 编码器 -> 特征
      - 3D xyz: 通过 Point_MAE 提取 token 特征
      - CSS: 在 2D & ps 特征之间做 Channel-Space Swap
      - Cross-Attn: 2D 与 3D 特征之间做双向 cross-attention
      - Decoder: 重建 RGB 图像和 depth map
    """

    def __init__(self,
                 rgb_backbone_name="vit_base_patch8_224_dino",
                 xyz_backbone_name="Point_MAE",
                 group_size=128,
                 num_group=1024,
                 img_size=224):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        # 复用你原来的 Model：里头有 ViT + Point_MAE
        self.encoder = Model(
            device=self.device,
            rgb_backbone_name=rgb_backbone_name,
            xyz_backbone_name=xyz_backbone_name,
            group_size=group_size,
            num_group=num_group,
        )

        # ==========================================
        # ★ 新增: Phase 1 - Shared Basis
        # ==========================================
        # num_points 对应 num_group (即 FPS 采样点数 M=1024)
        self.shared_basis = SharedBasis(num_points=num_group, k=16)

        # -------------------------
        # 2D / ps 分支的特征维度
        # -------------------------
        self.rgb_feat_dim = 768    # ViT-base 的 embedding 维度
        self.rgb_feat_hw = 28 * 28  # 224 / 8 = 28

        # 使用你自己的 CSS：2D & ps 的通道+空间交换
        self.css = CombinedExchange(p=2)

        # -------------------------
        # 3D 分支的特征维度 (Point_MAE)
        # xyz_backbone 输出维度为 1152 (3 层 * 384)
        # -------------------------
        self.xyz_token_dim = 1152
        # 为了让 2D 和 3D 特征能做 cross-attn，把 3D token 映射到与 2D 相同维度
        self.xyz_proj = nn.Linear(self.xyz_token_dim, self.rgb_feat_dim)
        # ==========================================
        # ★ 新增：终极修复 1 —— 升维投影层
        # 用于将解码器输出的 768 维特征升维回 1152 维，直面最纯正的物理几何真值！
        # ==========================================
        # self.xyz_out_proj = nn.Linear(self.rgb_feat_dim, self.xyz_token_dim)
        # ★ 终极修改 1：将预测目标直接变为 3 维物理坐标 (X, Y, Z)
        # self.xyz_out_proj = nn.Linear(self.rgb_feat_dim, 3)
        
        # ==========================================
        # ★ 完美架构：双头预测 (Dual-Head)
        # 1. 语义头：负责重构 1152 维 PointMAE 特征，抓划痕
        self.xyz_feat_head = nn.Linear(self.rgb_feat_dim, self.xyz_token_dim)
        # 2. 坐标头：负责重构 3 维绝对物理坐标，抓形变
        self.xyz_coord_head = nn.Linear(self.rgb_feat_dim, 3)
        # ==========================================
        
        # Step 2.2: 潜在空间随机掩码
        # 维度要和你 PointMAE 投影后的维度一致 (768)
        self.masking_module = LatentRandomMasking(
            input_dim=self.rgb_feat_dim, # 768
            mask_ratio=0.75               # 文档要求 60%
        )

        self.spectral_transform = SpectralTransform()
        self.cbdg = DynamicContentGating(num_points=num_group)
        
        # ==========================================
        # ★ Step 3.3: 谱域自适应混合专家 (FD-MoE)
        # ==========================================
        # 配置: 768维, 4个专家, 每次选2个
        self.fd_moe = FD_MoE(
            channels=self.rgb_feat_dim, # 768
            num_experts=4, 
            top_k=2
        )

        # ★ Step 3.4 & 3.5: GGRM
        self.ggrm = GGRM(
            channels=self.rgb_feat_dim, # 768
            num_points=1024,            # M
            reduction=4                 # 通道缩放比
        )
        
        # ★ Phase 4: 空间互补分支
        self.spatial_branch = SpatialComplementBranch(channels=self.rgb_feat_dim)
        
        # ★ Phase 5: 融合模块
        self.fusion = DualStreamFusion(channels=self.rgb_feat_dim)
        
        # LapWaveGate：3D 几何高频增强
        # self.lapwave = LapWaveGate(
        #     in_channels=3,          # xyz 坐标
        #     k=16,
        #     lambda_edge=0.1,
        #     unet_out_channels=128,  # 输出 Cg
        #     alpha=1.0,
        #     gamma=0.6,
        #     s_list=(0.25, 0.5, 1.0, 2.0),
        #     K=8,
        #     alpha_w=1.2,
        #     gate_a=8.0,
        #     gate_tau_init=0.5,
        #     beta=0.4,
        # )

        # # LapWave 输出通道到 Point_MAE 输入的映射
        # self.lap2xyz = nn.Linear(128, 3)   # Cg=128 -> 3 维坐标增量 Δxyz

        # # LapWaveGate 的损失权重（等会在 ReconFeatures 里用）
        # self.lap_loss_weight = 1e-4

        # -------------------------
        # 双向 Cross-Attention：2D ↔ 3D
        # -------------------------
        self.bdca = BiDirectionalCrossAttention(
            dim_2d=self.rgb_feat_dim,     # 2D token 维度
            dim_3d=self.rgb_feat_dim,     # 3D token 维度（投影后）
            num_heads=8,
            drop=0.1,
            order="2d_first",
        )

        # -------------------------
        # Decoder：从融合后的特征重建 RGB 和 depth
        # -------------------------
        # RGB 解码：从 (B, C, 28, 28) -> (B, 3, 224, 224)
        # self.rgb_decoder = nn.Sequential(
        #     nn.Conv2d(self.rgb_feat_dim, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 28 -> 112
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 112 -> 224
        #     nn.Conv2d(128, 3, kernel_size=3, padding=1),
        #     nn.Sigmoid(),
        # )

        # depth 解码：从 (B, C, 8, 8) -> (B, 1, 224, 224)
        # 这里假定 num_group=64，可 reshape 为 8x8 的粗略“空间”
        # self.xyz_decoder = nn.Sequential(
        #     nn.Conv2d(self.rgb_feat_dim, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 8 -> 32
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=7, mode="bilinear", align_corners=False),  # 32 -> 224
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1),
        # )
        # 4. 新的重建网络 (根据文档)
        # self.recon_2d = ReconNet2D(in_channels=self.rgb_feat_dim)
        # 计算特征图尺寸: 224 / 8 = 28
        feat_size = img_size // 8 
        
        self.recon_2d = ReconNet2D(
            in_channels=self.rgb_feat_dim, # 768
            embed_dim=96,                  # 内部通道数，Swin-Tiny标准是96，可调整
            img_size=feat_size,            # 显式传入 28
            window_size=7                  # 28 能被 7 整除，没问题
        )
        self.recon_3d = ReconNet3D(in_dim=self.rgb_feat_dim)
        
        
        # ★ 新增：用于永久存储训练集正常样本的像素级“误差基线”
        self.register_buffer('err_2d_mean', torch.zeros(1, img_size, img_size))
        self.register_buffer('err_2d_std', torch.ones(1, img_size, img_size))
        self.register_buffer('err_3d_mean', torch.zeros(1, img_size, img_size))
        self.register_buffer('err_3d_std', torch.ones(1, img_size, img_size))
        
        # ★ 新增：用于存储物理深度图的绝对基线
        self.register_buffer('depth_mean', torch.zeros(1, 1, img_size, img_size))
        self.register_buffer('depth_std', torch.ones(1, 1, img_size, img_size))
        
        # ==========================================
        # ★ 终极修复：必须冻结 Backbone 防止特征坍缩！
        # ==========================================
        for param in self.encoder.rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.encoder.xyz_backbone.parameters():
            param.requires_grad = False
            
        # 强制它们处于 eval 模式，防止 BatchNorm/Dropout 干扰提取出的纯净特征
        self.encoder.rgb_backbone.eval()
        self.encoder.xyz_backbone.eval()

    def forward(self, rgb, xyz,depth_map, ps):
        """
        rgb: (B, 3, 224, 224)
        ps : (B, 3, 224, 224)
        xyz: 
        - (B, 3, N)  组织点云被拉平成一维，N=H*W=224*224
        - 或 (B, 3, H, W) 原始网格
        """
        
        # 确保每次前向传播 Backbone 都在 eval 模式
        self.encoder.rgb_backbone.eval()
        self.encoder.xyz_backbone.eval()
        
        B = rgb.size(0)

        # -------------------------
        # 1) 2D & ps 特征提取（ViT）
        # -------------------------
        rgb_feats = self.encoder.forward_rgb_features(rgb)  # (B, 768, 28, 28)
        ps_feats  = self.encoder.forward_rgb_features(ps)   # (B, 768, 28, 28)

        # -------------------------
        # 2) CSS：2D & ps 之间做通道+空间交换
        # -------------------------
        rgb_feats_css, ps_feats_css = self.css(rgb_feats, ps_feats)
        # 后面重建 & cross-attn 用 rgb_feats_css，ps_feats_css 留给后续模块也可以
       
       # ==========================================
        # ★ 终极底牌：去噪自编码器 (Denoising Autoencoder)
        # 注入高斯噪声！既破坏了网络的“抄答案”能力，又保留了完整的空间锚点供 3D 参考。
        # 网络必须学会从噪声中“猜”出正常的零件，遇到缺陷时根本猜不出，从而爆红！
        # ==========================================
        # if self.training:
        #     # 注入标准差为 0.15 的高斯扰动
        #     noise = torch.randn_like(rgb_feats_css) * 0.15
        #     rgb_feats_bottleneck = rgb_feats_css + noise
        # else:
        #     rgb_feats_bottleneck = rgb_feats_css
        rgb_feats_bottleneck = rgb_feats_css.detach()
       
       
       # 兼容 xyz 输入格式
        if xyz.dim() == 4:
             # (B, 3, H, W) -> (B, N, 3)
            B, C, H, W = xyz.shape
            xyz = xyz.view(B, C, -1).transpose(1, 2).contiguous()


        # ==========================================
        # ★ 终极点云唤醒术：FPS 坐标坍缩 + 单位球归一化
        # ==========================================
        # 1. 获取前景掩码 (用你之前的 batch_min_z 方法)
        depth_for_mask = ps.mean(dim=1, keepdim=True) # 这里请换成你实际获取深度图的代码
        batch_min_z = depth_for_mask.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        fg_mask_224 = (depth_for_mask > batch_min_z + 1e-5).float()
        fg_mask_flat = fg_mask_224.view(B, -1, 1) # [B, 50176, 1]
        
        # 2. 坐标坍缩：背景坐标全变为 0，强迫 FPS 100% 采前景
        xyz_collapsed = xyz * fg_mask_flat

        # 3. 计算前景工件的物理质心 (Center of Mass)
        # 加 1e-5 防止除以 0
        valid_count = fg_mask_flat.sum(dim=1, keepdim=True) + 1e-5 
        center_of_mass = xyz_collapsed.sum(dim=1, keepdim=True) / valid_count # [B, 1, 3]
        
        # 4. 去中心化：把工件移动到 (0,0,0) 原点
        xyz_centered = (xyz_collapsed - center_of_mass) * fg_mask_flat
        
        # 5. 极值缩放：强制压缩到 [-1, 1] 的单位球内
        max_dist = torch.norm(xyz_centered, dim=-1, keepdim=True).max(dim=1, keepdim=True)[0] + 1e-5
        xyz_normalized = (xyz_centered / max_dist) * fg_mask_flat
            
            
        # -------------------------
        # 4) 3D 编码：Point_MAE
        # -------------------------
        # print("xyz.shape:",xyz.shape)
        xyz_tokens, center, ori_idx, center_idx = self.encoder.xyz_backbone(xyz_normalized)
        # ==========================================
        # ★ 新增: Phase 1 (Step 1.2 - 1.4)
        # 输入: center (采样后的几何坐标)
        # 输出: U (基底), loss_geo (几何正则化损失)
        # ==========================================
        U, loss_geo, knn_idx = self.shared_basis(center, ps, center_idx)
        
        # print("U.shape:",U.shape)
        
        # xyz_tokens: (B, 1152, G)  G = num_group = 64
        xyz_tokens = xyz_tokens.transpose(1, 2).contiguous()          # (B, G, 1152)
        xyz_tokens = torch.nan_to_num(xyz_tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        # ==========================================
        # ★ 终极修复 2：激活 3D 几何推断（摧毁死区）
        # 直接拿 PointMAE 提取的、富含绝对空间形变信息的 1152 维特征作为真值 (F_gt)！
        # ==========================================
        # F_gt = xyz_tokens.detach() 
        # ==========================================
        # ★ 终极修改 2：彻底抛弃语义特征，直接拿物理采样点 center (B, 1024, 3) 作为绝对真值！
        # ==========================================
        # F_gt = center.detach()
        
        # ★ 提取双真值 (都 detach 切断梯度)
        F_feat_gt = xyz_tokens.detach()   # 1152 维语义真值
        F_coord_gt = center.detach()      # 3 维坐标真值
        # print("Target STD:", F_feat_gt.std(dim=1).mean().item())
        # print("F_feat_gt.shape:",F_feat_gt.shape)
        
        # 投影到 768 维度，作为网络内部处理（Mask 和 频域变换）的起点
        F_in_raw = self.xyz_proj(xyz_tokens.detach())

        # ==========================================
        # ★ Step 2.2: 潜在空间随机掩码
        # 输入: F_gt
        # 输出: F_in (masked), mask (binary)
        # ==========================================
        if self.training:
            # 训练时：执行掩码
            F_in, mask = self.masking_module(F_in_raw)
        else:
            # 测试/推理时：通常不掩码，或者是为了做修复任务而掩码？
            # 工业异常检测的常见做法是：推理时也掩码，看模型能不能修补回来。
            # 如果你的逻辑是“重构误差”，那么测试时也需要掩码。
            # 根据 PointMAE 原理，测试时通常也进行掩码重建。
            F_in, mask = self.masking_module(F_in_raw)
            
            # 如果你想测试时全量输入(不掩码)，可以用:
            # F_in = F_in_raw
            # mask = torch.zeros(B, F_in_raw.shape[1], device=self.device)
        
        # ==========================================
        # ★ Phase 3 & 4 接口准备
        # ==========================================
        # 现在的 F_in 就是文档中 Step 3.1 的输入
        # F_in shape: [B, 1024, 768] (包含 [MASK] token)
        # U: [B, 1024, 1024], F_in: [B, 1024, 768]
        F_spec = self.spectral_transform.gft(F_in, U)  # -> [B, 1024, 768]
        # F_low, F_high 都是 [B, 1024, 768]
        # 但 F_low 主要保留了前段频率，F_high 保留了后段
        F_low, F_high, gate_map = self.cbdg(F_spec)
        
        # ==========================================
        # ★ Step 3.3: FD-MoE 处理低频分支
        # ==========================================
        # 输入: F_low (频域特征), U (用于内部逆变换)
        # 输出: F_low_spatial (修复后的完美几何特征，已变回空间域)
        F_low_spatial = self.fd_moe(F_low, U)
        
        # ==========================================
        # ★ Step 3.4 & 3.5: GGRM 处理高频并重组
        # ==========================================
        # 输入:
        # 1. Guide: F_low_spatial (空间域)
        # 2. Target: F_high (谱域, GGRM 内部会做逆变换)
        # 3. U: 正交基底
        xyz_features_processed = self.ggrm(F_low_spatial, F_high, U)
        
        # ==========================================
        # ★ Phase 4: 空间互补分支
        # 输入: F_in (掩码后的特征), knn_idx (完全相同的几何拓扑)
        # ==========================================
        F_spatial_out = self.spatial_branch(F_in, knn_idx)
        
        # ==========================================
        # ★ Phase 5: 融合与重建
        # ==========================================
        # 输入:
        # 1. 频域特征 (F_freq_out)
        # 2. 空间特征 (F_spatial_out)
        # 3. 原始输入 (F_in, 含 Mask Token, 用于残差连接)

        xyz_recon_final = self.fusion(
            f_freq_out=xyz_features_processed, 
            f_spatial_out=F_spatial_out, 
            f_in=F_in
        )
        
        # -------------------------
        # 5) 2D 特征 flatten 成 token 序列
        # -------------------------
        # B2, C_rgb, H_rgb, W_rgb = rgb_feats_css.shape             # C_rgb = 768, H=W=28
        # ★ 注意：这里使用的是带瓶颈的特征 rgb_feats_bottleneck！
        B2, C_rgb, H_rgb, W_rgb = rgb_feats_bottleneck.shape
        N_rgb = H_rgb * W_rgb

        # (B, 768, 28, 28) -> (B, 784, 768)
        # rgb_tokens = rgb_feats_css.view(B2, C_rgb, N_rgb).permute(0, 2, 1)
        rgb_tokens = rgb_feats_bottleneck.view(B2, C_rgb, N_rgb).permute(0, 2, 1)

        # -------------------------
        # 6) 双向 Cross-Attention：2D ↔ 3D
        # -------------------------
        rgb_updated, xyz_updated = self.bdca(rgb_tokens, xyz_recon_final)
        # ★ 黄金验证：强行切断 2D 给 3D 递小抄的通道！
        # rgb_updated = rgb_tokens
        # xyz_updated = xyz_recon_final
        # rgb_updated: (B, 784, 768)
        # xyz_updated: (B, G,  768)
        
        # -------------------------
        # 8) 解码重建
        # -------------------------
        # 2D 重建
        # Reshape back: [B, 768, 28, 28]
        # rgb_in_recon = rgb_updated.permute(0, 2, 1).view(B, C, H, W)
        B, N, D = rgb_updated.shape          # (B, 784, 768)
        h = w = int(N ** 0.5)               # 28
        assert h * w == N, (N, h, w)
        rgb_in_recon = rgb_updated.permute(0, 2, 1).contiguous().reshape(B, D, h, w)  # (B,768,28,28)
        # Output: [B, 768, 28, 28] -> Frgb (reconstructed)
        rgb_recon_feat = self.recon_2d(rgb_in_recon)
        # 3D 重建
        # Input: [B, G, 768] -> Output: [B, G, 768] -> Fpred
        xyz_recon_feat = self.recon_3d(xyz_updated,center)
        
        # ==========================================
        # ★ 终极修复 3：映射回 1152 维去计算损失
        # ==========================================
        # xyz_recon_final_out = self.xyz_out_proj(xyz_recon_feat)
        
        # ★ 分别通过两个头输出
        xyz_recon_feat_out = self.xyz_feat_head(xyz_recon_feat)   # [B, 1024, 1152]
        xyz_recon_coord_out = self.xyz_coord_head(xyz_recon_feat) # [B, 1024, 3]
        
        
        # # ==================== 泄漏探针开始 ====================
        # print("\n--- 3D 泄漏排查探针 ---")
        # # 1. 检查 Mask 极性是否反转
        # # 假设 N=1024, Mask_ratio=0.75。如果打印出 ~256，说明你算 Loss 算在了【可见点】上！可见点当然误差为0！
        # print(f"[探针 1] Mask 点数 (应该是 ~{int(xyz_tokens.shape[1] * 0.75)}):", mask[0].sum().item())
        
        # # 2. 检查 Masking 模块是否失效
        # # 如果 F_in 和 F_in_raw 是一样的，说明你的 masking_module 根本没有把特征替换成 Mask Token！
        # print("[探针 2] F_in 是否完全等于未掩码的 F_in_raw? :", torch.allclose(F_in, F_in_raw, atol=1e-6))
        
        # # 3. 检查 Mask Token 的纯度
        # # 真正的 Mask 区域，特征应该全部被替换成了同一个可学习的 mask_token，所以它的空间方差应该是 0.0！
        # # 如果打印出来不是 0.0，说明掩码区域依然带着原始信息！
        # mask_bool = mask[0].bool() # 拿第一个 batch 的 mask
        # if mask_bool.sum() > 0:
        #     print("[探针 3] Mask 区域内部的特征方差 (必须是 0.0!):", F_in[0, mask_bool, :].std(dim=0).mean().item())
            
        # # 4. 检查预测结果是不是直接复制了原始特征
        # # 如果网络什么都没做，直接把 F_in_raw 传过来了，这里的相似度会极高
        # print("[探针 4] 预测特征与 F_in_raw 的绝对差异:", (xyz_recon_feat - F_in_raw).abs().mean().item())
        # print("-----------------------\n")
        # # ==================== 泄漏探针结束 ====================
        
        return rgb_recon_feat, xyz_recon_feat_out, xyz_recon_coord_out, rgb_feats_css, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx


def splat_3d_error_to_2d_exact(errors, center_idx, img_size):
    B, N = errors.shape
    
    # 1. 直接在 224x224 的绝对分辨率画布上初始化
    error_map = torch.zeros((B, img_size * img_size), device=errors.device)
    
    # 过滤掉无效点 (center_idx == 0 的 padding 区域)
    valid_mask = (center_idx > 0).float()
    errors_clean = errors * valid_mask
    
    # 2. ★ 核心突围：直接映射绝对位置，坚决不求平均！
    # 将稀疏的误差峰值像钉钉子一样原封不动地砸在 224x224 画布上
    error_map.scatter_(1, center_idx.long(), errors_clean)
    error_map = error_map.view(B, 1, img_size, img_size)
    
    # 3. ★ 峰值膨胀连通 (Dilation)
    # 因为 1024 个点在 224x224 上是非常稀疏的星空状，
    # 使用 MaxPool 可以把那 1-2 个极亮的高分点直接扩张成一个红色的区块，而不被周围的 0 值稀释。
    # kernel_size=11 确保 1024 个点扩张后能无缝覆盖整个工件表面
    error_map_dilated = F.max_pool2d(error_map, kernel_size=11, stride=1, padding=5)
    
    # 4. 轻柔化边缘，消除方块马赛克感
    error_map_smooth = F.avg_pool2d(error_map_dilated, kernel_size=5, stride=1, padding=2)
    
    return error_map_smooth.squeeze(1)

# ------------------------------------------------
# 3. 重建式异常检测封装
# ------------------------------------------------
class ReconFeatures(nn.Module):
    """
    重建式异常检测：
      - train_step: 只算重建 loss (RGB + depth)
      - predict   : 输出融合重建误差 heatmap
    """

    def __init__(self, args):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.img_size_val = getattr(args, 'img_size', 224)
        self.net = FusionReconNet(
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group,
            img_size=args.img_size_val,
        ).to(self.device)

        self.blur = KNNGaussianBlur(2)
        
        # 超参数
        # l2_weight: 辅助损失的权重，通常设小一点 (e.g. 0.1 或 0.5)
        self.l2_weight = 0.5
        self.lambda_geo = getattr(args, 'lambda_geo', 0.1) # 从 args 读取，默认 0.1
        
        # ★ OT 模块 (用于训练 Loss 平衡)
        self.ot_module = UncertaintyAwareOT(momentum=0.9, init_temperature=0.2).to(self.device)
        
        # 指标与可视化缓存（仿 RealIAD-D3 的 Method 风格）
        self.reset_buffers()
        
        
        # ★ 终极调参中心：推理时控制三个模态得分上限的平衡系数
        self.weight_2d = 1.0           # 2D 基础权重
        self.weight_3d_feat = 2.0      # 3D 语义压制系数
        self.weight_depth = 0.3        # 物理深度压制系数 (因为方差极小，极易爆炸，必须重度压制)

    def reset_buffers(self):
        #   """清空评估阶段用到的缓存（仿 RealIAD-D3 的 Method）"""
        # 图像级
        self.image_preds = []   # 每张图的异常分数（如 max fmap）
        self.image_labels = []  # 每张图的标签 0/1

        # 像素级 (注意内存，如果显存/内存不足，不要存 pixel_preds)
        self.pixel_preds = []   # 所有样本的像素级异常分数
        self.pixel_labels = []  # 所有样本的像素级 GT (0/1)

        # 方便算 AU-PRO：每张图的 GT 和 Pred map
        self.gts = []           # list[np.ndarray]，每个 [H,W]
        self.pred_maps = []     # list[torch.Tensor] or np.ndarray，[H,W]

        # 可视化需要的原始输入（rgb, depth, ps）
        self.vis_samples = []   # list[(rgb_b, depth_b, ps_b)]
        
        
        # ★ 新增：初始化用于缓存可视化热力图的空列表
        self.maps_2d = [] 
        self.maps_3d = []

    def compute_hybrid_loss(self, pred, target, dim=-1, mask=None):
        """
        混合损失函数：强制 L2 归一化防坍缩
        """
        pred_safe = pred + 1e-8
        target_safe = target + 1e-8
        
        # 1. Cosine Loss
        cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim)
        
        # 2. L2 Loss (★ 核心修复：必须在 L2 归一化后计算，防止模型靠缩短向量长度作弊)
        pred_norm = F.normalize(pred, p=2, dim=dim)
        target_norm = F.normalize(target, p=2, dim=dim)
        
        l2_loss_raw = F.mse_loss(pred_norm, target_norm, reduction='none')
        
        if dim == 1: # 2D: [B, H, W]
            l2_loss = l2_loss_raw.mean(dim=1)
        else:        # 3D: [B, N]
            l2_loss = l2_loss_raw.mean(dim=-1)
            
        total_loss_map = cosine_loss + self.l2_weight * l2_loss
        
        # 3. 掩码过滤
        if mask is not None:
            total_loss_map = total_loss_map * mask
            return total_loss_map.sum() / (mask.sum() + 1e-5)
        else:
            return total_loss_map.mean()

    # -----------------------
    # 训练一步
    # -----------------------
    def train_step(self, sample):
        """
        sample: (rgb, xyz, depth_map, ps)
          - rgb        : (B, 3, 224, 224)
          - xyz        : (B, 3, N)
          - depth_map  : (B, 3, 224, 224)  (RealIAD 里的 z-channel 已经复制成3通道)
          - ps         : (B, 3, 224, 224)
        """
        rgb, xyz, depth_map, ps = sample

        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        ps = ps.to(self.device)

        # Forward
        # rgb_recon: 重建后的2D特征 [B, 768, 28, 28]
        # xyz_recon: 重建后的3D特征 [B, 1024, 768]
        # rgb_target: CSS后的2D特征 [B, 768, 28, 28]
        # xyz_target: 投影后的3D真值特征 [B, 1024, 768]
        # 注意这里精准解包了所有的 11 个返回值
        rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx = self.net(rgb, xyz, depth_map, ps)

        # ==========================================
        # ★ 核心大招：生成训练专用的 2D 损失掩码 (Loss Masking)
        # ==========================================
        depth_for_mask = depth_map.to(self.device)
        if depth_for_mask.dim() == 4:
            if depth_for_mask.shape[1] == 3: depth_for_mask = depth_for_mask[:, 2, :, :]
            elif depth_for_mask.shape[-1] == 3: depth_for_mask = depth_for_mask[:, :, :, 2]
            else: depth_for_mask = depth_for_mask.mean(dim=1)
        if depth_for_mask.dim() == 3:
            depth_for_mask = depth_for_mask.unsqueeze(1)
            
        batch_min_z = depth_for_mask.view(rgb.size(0), -1).min(dim=1)[0].view(rgb.size(0), 1, 1, 1)
        fg_mask_224 = (depth_for_mask > batch_min_z + 1e-5).float()
        
        # 向内腐蚀边缘 (kernel=15)，告诉网络：绝对不要去重建边缘！
        bg_mask = 1.0 - fg_mask_224
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=15, stride=1, padding=7)
        train_mask_224 = 1.0 - eroded_bg
        
        # 下采样到 28x28，适配 2D 特征图的大小
        train_mask_28 = F.interpolate(train_mask_224, size=(28, 28), mode='nearest').squeeze(1) # [B, 28, 28]

        # -------------------------
        # 计算 Loss
        # -------------------------
        
        # 1. 2D Loss: MSE(Frgb, Frgb')
        # 文档: "2d分支的损失函数是 Frgb 和 Frgb, 之间的误差"
        # 2D (Channel维度是 dim=1)
        # loss_2d = self.compute_hybrid_loss(rgb_recon, rgb_target.detach(),dim = 1) 
        loss_2d = self.compute_hybrid_loss(rgb_recon, rgb_target.detach(),dim = 1, mask=train_mask_28) 
        # 注意: 通常 Target 不需要梯度，detach 掉 rgb_target 以防止梯度传回 Encoder
        
        # 2. 3D Loss: Global MSE Loss
        # 文档: "Lrec = |Fpred - Fgt|^2"
        # 3D (Channel维度是 dim=2)
        # loss_3d = self.compute_hybrid_loss(xyz_recon, F_gt.detach(),dim = 2)
        # loss_3d = self.compute_hybrid_loss(xyz_recon, F_gt.detach(),dim = 2, mask=mask)
        # ==========================================
        # ★ 终极修改 3：3D Loss 变为纯粹的物理距离误差 (欧氏距离平方)
        # ==========================================
        # l2_loss_3d_raw = F.mse_loss(xyz_recon, F_gt.detach(), reduction='none') # [B, 1024, 3]
        # l2_loss_3d = l2_loss_3d_raw.mean(dim=-1) # [B, 1024]
        # loss_3d = (l2_loss_3d * mask).sum() / (mask.sum() + 1e-5)
        
        # 2. 3D 语义 Loss (带 F.normalize，防止 0.003 坍缩)
        # ==========================================
        # ★ 终极修复 1：提取 3D 点云的前景掩码
        # 将 2D 掩码展平，利用 center_idx 找出 1024 个点中哪些属于工件！
        # ==========================================
        B = rgb.size(0)
        fg_mask_flat = fg_mask_224.view(B, -1) # [B, 50176]
        fg_mask_3d = torch.gather(fg_mask_flat, 1, center_idx.long()) # [B, 1024]
        
        # 真正的有效掩码 = 被随机 Mask 遮挡的区域 AND 必须是工件前景！
        effective_3d_mask = mask * fg_mask_3d
        
        
        
        # 用有效掩码计算 3D Loss
        loss_3d_feat = self.compute_hybrid_loss(xyz_feat_recon, F_feat_gt, dim=2, mask=effective_3d_mask)
        
        # ==========================================
        # ★ 终极杀手锏：纯 Z 轴物理坐标监督 (完全抛弃XY死网格)
        # ==========================================
        pred_z = xyz_recon_coord_out[:, :, 2]   # [B, 1024]
        gt_z = F_coord_gt[:, :, 2].detach()     # [B, 1024]
        
        loss_3d_coord_raw = F.l1_loss(pred_z, gt_z, reduction='none') 
        loss_3d_coord = (loss_3d_coord_raw * effective_3d_mask).sum() / (effective_3d_mask.sum() + 1e-5)
        
        # 50.0 权重给纯粹的 Z 轴误差提权
        loss_3d = loss_3d_feat + 10.0 * loss_3d_coord
        # loss_3d = loss_3d_feat
        # loss_3d_feat = self.compute_hybrid_loss(xyz_feat_recon, F_feat_gt, dim=2, mask=mask)
        # loss_3d = loss_3d_feat 
        
        # # 3. ★ OT 计算权重 (传入 detach 防止作弊)
        # # 只有在训练时，我们希望"避重就轻"，让模型先学容易的，稳步收敛
        # alpha, beta = self.ot_module(loss_2d.detach(), loss_3d.detach())
        
        # # 取 Batch 平均权重进行反向传播
        # w_alpha = alpha.mean()
        # w_beta = beta.mean()
        
        # ---------- 换用绝对稳定的固定权重 ----------
        # 因为 2D Loss 稍微大一点点，1:1 或者 1:2 都是极佳的，这里推荐 1:1 稳如泰山
        w_alpha = 1.0
        w_beta = 1.0
    
        weighted_loss = w_alpha * loss_2d + w_beta * loss_3d
        total_loss = weighted_loss + self.lambda_geo * loss_geo
        
        return {
            "loss": total_loss,
            "l2d": loss_2d.item(),
            "l3d": loss_3d.item(),
            # ★ 新增：输出未乘系数的纯净版 3D 子 Loss
            "l3d_feat": loss_3d_feat.item(), 
            "l3d_coord": loss_3d_coord.item(),
            "geo": loss_geo.item(),
            "alpha": w_alpha,
            "beta": w_beta
        }
        
    @torch.no_grad()
    def build_error_statistics(self, train_loader):
        """
        在训练的最后一个 Epoch 结束后调用。
        遍历一次训练集，提取所有正常样本的平均重建误差和波动标准差，建立纯净基线。
        """
        self.net.eval()
        err_2d_list = []
        # err_3d_list = []
        err_3d_feat_list = []
        depth_list = []
        
        
        from tqdm import tqdm
        print("\n[Post-Training] Building Pixel-wise Z-Score Baseline...")
        for sample, _ in tqdm(train_loader, desc="Extracting Baseline"):
            rgb, xyz, depth_map, ps = sample
            rgb, xyz, ps = rgb.to(self.device), xyz.to(self.device), ps.to(self.device)
            B = rgb.size(0)

            rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, _, _, _, _, center_idx = self.net(rgb, xyz, depth_map, ps)
            # 1. 算 Raw 误差
            # === 彻底替换掉这 6 行旧代码 ===
            rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)
            rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
            # xyz_recon_norm = F.normalize(xyz_recon, p=2, dim=2)
            # F_gt_norm = F.normalize(F_gt, p=2, dim=2)

            # err_2d = torch.sum((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
            # if err_2d.shape[-1] != self.img_size_val:
            #     err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

            # err_3d_points = torch.sum((xyz_recon_norm - F_gt_norm) ** 2, dim=2)
            
            # === 彻底替换为真正的混合误差算子：余弦方向误差 + 绝对幅值误差 ===
            
            # 2D 误差图计算 (不归一化，保留绝对幅值)
            err_2d = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                     self.l2_weight * torch.mean((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
            
            if err_2d.shape[-1] != self.img_size_val:
                err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

            # ★ 3D 误差双重暴击
            # A. 语义误差：抓划痕
            xyz_feat_norm = F.normalize(xyz_feat_recon, p=2, dim=2)
            F_feat_gt_norm = F.normalize(F_feat_gt, p=2, dim=2)
            err_3d_feat = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                          self.l2_weight * torch.mean((xyz_feat_norm - F_feat_gt_norm) ** 2, dim=2)
            
            # ★ 加入纯 Z 轴基线计算
            pred_z = xyz_recon_coord_out[:, :, 2]
            gt_z = F_coord_gt[:, :, 2]
            err_3d_coord = torch.abs(pred_z - gt_z)
            
            total_3d_err = err_3d_feat + 10.0 * err_3d_coord
            err_3d_map = splat_3d_error_to_2d_exact(total_3d_err, center_idx, self.img_size_val)
            
            # err_3d_map = splat_3d_error_to_2d_exact(err_3d_feat, center_idx, self.img_size_val)

            # 3. 提取物理深度真值 (★ 为模板匹配建立基线)
            depth_gt = depth_map.to(self.device)
            if depth_gt.dim() == 4:
                if depth_gt.shape[1] == 3: depth_gt = depth_gt[:, 2:3, :, :]
                elif depth_gt.shape[-1] == 3: depth_gt = depth_gt[:, :, :, 2:3]
                else: depth_gt = depth_gt.mean(dim=1, keepdim=True)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)
                
            err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).squeeze(1)
            err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).squeeze(1)

            err_2d_list.append(err_2d_smooth)
            err_3d_feat_list.append(err_3d_smooth)
            depth_list.append(depth_gt.cpu())
            
        all_err_2d = torch.cat(err_2d_list, dim=0) # [Total_Samples, H, W]
        all_err_3d_feat = torch.cat(err_3d_feat_list, dim=0)
        all_depth = torch.cat(depth_list, dim=0)

        # 3. 注入模型的“记忆”中
        self.net.err_2d_mean.copy_(all_err_2d.mean(dim=0, keepdim=True))
        self.net.err_2d_std.copy_(all_err_2d.std(dim=0, keepdim=True) + 1e-5)
        self.net.err_3d_mean.copy_(all_err_3d_feat.mean(dim=0, keepdim=True))
        self.net.err_3d_std.copy_(all_err_3d_feat.std(dim=0, keepdim=True) + 1e-5)
        self.net.depth_mean.copy_(all_depth.mean(dim=0, keepdim=True))
        self.net.depth_std.copy_(all_depth.std(dim=0, keepdim=True) + 1e-5)
        print("Baseline and Physical Depth Template established and locked!")


    # -----------------------
    # 推理：累积式预测（仿 RealIAD-D3 的 Method）
    # -----------------------
    @torch.no_grad()
    def predict(self, sample, gt, label, rgb_path=None):
        if not isinstance(sample, (tuple, list)):
            raise RuntimeError(f"ReconFeatures.predict 期望 sample 是 tuple/list，实际类型: {type(sample)}")

        rgb, xyz, depth_map, ps = sample
        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        depth_map = depth_map.to(self.device)
        ps = ps.to(self.device)
        B = rgb.size(0)
        
        ensemble_runs = 4  
        err_3d_total_ensemble = 0

        for _ in range(ensemble_runs):
            rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, _, _, _, _, center_idx = self.net(rgb, xyz, depth_map, ps)
            
            xyz_feat_norm = F.normalize(xyz_feat_recon, p=2, dim=2)
            F_feat_gt_norm = F.normalize(F_feat_gt, p=2, dim=2)
            
            err_3d_feat_current = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                                  self.l2_weight * torch.mean((xyz_feat_norm - F_feat_gt_norm) ** 2, dim=2)
            
            # ★ 仅依赖 Z 轴下陷误差进行微小缺陷判定
            pred_z = xyz_recon_coord_out[:, :, 2]
            gt_z = F_coord_gt[:, :, 2]
            err_3d_coord_current = torch.abs(pred_z - gt_z)
            
            err_3d_total_ensemble += (err_3d_feat_current + 10.0 * err_3d_coord_current)

        err_3d_total = err_3d_total_ensemble / ensemble_runs

        rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)   
        rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
        
        err_2d = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                 self.l2_weight * torch.mean((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
                 
        if err_2d.shape[-1] != self.img_size_val:
            err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

        err_3d_map = splat_3d_error_to_2d_exact(err_3d_total, center_idx, self.img_size_val)

        depth_for_eval = depth_map.to(self.device)
        if depth_for_eval.dim() == 4:
            if depth_for_eval.shape[1] == 3: depth_for_eval = depth_for_eval[:, 2:3, :, :]
            elif depth_for_eval.shape[-1] == 3: depth_for_eval = depth_for_eval[:, :, :, 2:3]
            else: depth_for_eval = depth_for_eval.mean(dim=1, keepdim=True)
        if depth_for_eval.dim() == 3:
            depth_for_eval = depth_for_eval.unsqueeze(1)
            
        batch_min_z = depth_for_eval.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        fg_mask_raw = (depth_for_eval > batch_min_z + 1e-5).float()
        
        err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).to(self.device)
        err_2d_smooth = err_2d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).to(self.device)
        err_3d_smooth = err_3d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        std_2d = self.net.err_2d_std.to(self.device)
        std_3d_feat = self.net.err_3d_std.to(self.device)
        
        # ==========================================
        # ★ 极其关键的硬底线防爆：镇压底层浮点噪声放大的红斑！
        # ==========================================
        eps_2d = torch.clamp(std_2d.mean() * 0.1, min=1e-2)
        eps_3d = torch.clamp(std_3d_feat.mean() * 0.1, min=1e-2) 
        
        err_2d_norm = F.relu(err_2d_smooth - self.net.err_2d_mean.to(self.device)) / (std_2d + eps_2d)
        err_3d_feat_norm = F.relu(err_3d_smooth - self.net.err_3d_mean.to(self.device)) / (std_3d_feat + eps_3d)

        std_depth = self.net.depth_std.to(self.device)
        eps_depth = torch.clamp(std_depth.mean() * 0.1, min=1e-2)
        
        err_depth_raw = torch.abs(depth_for_eval - self.net.depth_mean.to(self.device))
        err_depth_norm = err_depth_raw.squeeze(1) / (std_depth.squeeze(1) + eps_depth)

        fg_mask_dilated = F.max_pool2d(fg_mask_raw, kernel_size=7, stride=1, padding=3)
        fg_mask_eval = F.interpolate(fg_mask_dilated, size=(self.img_size_val, self.img_size_val), mode='nearest').squeeze(1)
        
        err_2d_norm = err_2d_norm * fg_mask_eval
        err_3d_feat_norm = err_3d_feat_norm * fg_mask_eval
        err_depth_norm = err_depth_norm * fg_mask_eval

        fused = self.weight_2d * err_2d_norm + \
                self.weight_3d_feat * err_3d_feat_norm + \
                self.weight_depth * err_depth_norm

        bg_mask = 1.0 - fg_mask_raw
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=5, stride=1, padding=2)
        score_mask = (1.0 - eroded_bg).squeeze(1).cpu()
        
        gt = gt.detach().cpu()
        label = label.detach().cpu()
        fused = fused.cpu()
        for b in range(B):
            gmap = gt[b, 0]           
            fmap = fused[b]          

            fmap_score = fmap * score_mask[b]
            fmap_flat = fmap_score.flatten()
            k_val = max(1, int(fmap_flat.numel() * 0.01)) 
            topk_vals, _ = torch.topk(fmap_flat, k_val)
            img_score = float(topk_vals.mean())
            
            self.image_preds.append(img_score)
            self.image_labels.append(int(label[b]))

            self.gts.append(gmap.numpy())
            self.pred_maps.append(fmap.clone())
            
            self.maps_2d.append(err_2d_norm[b].clone())
            self.maps_3d.append(err_3d_feat_norm[b].clone())

            if isinstance(sample, (tuple, list)):
                rgb_b   = sample[0][b:b+1].cpu()
                depth_b = sample[2][b:b+1].cpu()
                ps_b    = sample[3][b:b+1].cpu()
            else:
                rgb_b = depth_b = ps_b = None

            self.vis_samples.append((rgb_b, depth_b, ps_b))

        return

    def calculate_metrics(self):
        from sklearn.metrics import roc_auc_score
        from utils.au_pro_util import calculate_au_pro

        if len(self.image_labels) == 0:
            self.image_rocauc = float("nan")
            self.pixel_rocauc = float("nan")
            self.au_pro = float("nan")
            return

        if len(set(self.image_labels)) > 1:
            self.image_rocauc = float(roc_auc_score(self.image_labels, self.image_preds))
        else:
            self.image_rocauc = float("nan")

        gts_flat = np.concatenate([gt.flatten() for gt in self.gts])
        preds_flat = np.concatenate([pred.cpu().numpy().flatten() for pred in self.pred_maps])
        
        assert gts_flat.shape == preds_flat.shape, f"Shape mismatch: GT {gts_flat.shape} vs Pred {preds_flat.shape}"
        self.pixel_rocauc = float(roc_auc_score(gts_flat, preds_flat))# feature_extractors/recon_features.py
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 添加这行

from models.models import Model  # 复用原来的 2D/3D 编码器
from utils.utils import KNNGaussianBlur
from feature_extractors.change_ex import CombinedExchange   # ★ 新增：用你的 CSS 模块
from models.cross_attention import BiDirectionalCrossAttention  # ← 新增
from models.lapwavegate import LapWaveGate
from utils.ot_fusion import fuse_by_task_ot,robust_zscore
from feature_extractors.ReconNet2D import ReconNet2D
from feature_extractors.ReconNet3D import ReconNet3D
from feature_extractors.shared_basis import SharedBasis  # <--- 新增
from feature_extractors.masking import LatentRandomMasking
from feature_extractors.spectral_branch import *
from feature_extractors.spatial_branch import SpatialComplementBranch
from feature_extractors.fusion_spatial_spectral import DualStreamFusion
from feature_extractors.ot_module import UncertaintyAwareOT



# ------------------------------------------------
# 主干网络：2D / ps 编码 + CSS + 3D 编码 + Cross-Attn + 重建
# ------------------------------------------------
class FusionReconNet(nn.Module):
    """
    主干网络：
      - 2D RGB & ps: 先各自通过 ViT 编码器 -> 特征
      - 3D xyz: 通过 Point_MAE 提取 token 特征
      - CSS: 在 2D & ps 特征之间做 Channel-Space Swap
      - Cross-Attn: 2D 与 3D 特征之间做双向 cross-attention
      - Decoder: 重建 RGB 图像和 depth map
    """

    def __init__(self,
                 rgb_backbone_name="vit_base_patch8_224_dino",
                 xyz_backbone_name="Point_MAE",
                 group_size=128,
                 num_group=1024,
                 img_size=224):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        # 复用你原来的 Model：里头有 ViT + Point_MAE
        self.encoder = Model(
            device=self.device,
            rgb_backbone_name=rgb_backbone_name,
            xyz_backbone_name=xyz_backbone_name,
            group_size=group_size,
            num_group=num_group,
        )

        # ==========================================
        # ★ 新增: Phase 1 - Shared Basis
        # ==========================================
        # num_points 对应 num_group (即 FPS 采样点数 M=1024)
        self.shared_basis = SharedBasis(num_points=num_group, k=16)

        # -------------------------
        # 2D / ps 分支的特征维度
        # -------------------------
        self.rgb_feat_dim = 768    # ViT-base 的 embedding 维度
        self.rgb_feat_hw = 28 * 28  # 224 / 8 = 28

        # 使用你自己的 CSS：2D & ps 的通道+空间交换
        self.css = CombinedExchange(p=2)

        # -------------------------
        # 3D 分支的特征维度 (Point_MAE)
        # xyz_backbone 输出维度为 1152 (3 层 * 384)
        # -------------------------
        self.xyz_token_dim = 1152
        # 为了让 2D 和 3D 特征能做 cross-attn，把 3D token 映射到与 2D 相同维度
        self.xyz_proj = nn.Linear(self.xyz_token_dim, self.rgb_feat_dim)
        # ==========================================
        # ★ 新增：终极修复 1 —— 升维投影层
        # 用于将解码器输出的 768 维特征升维回 1152 维，直面最纯正的物理几何真值！
        # ==========================================
        # self.xyz_out_proj = nn.Linear(self.rgb_feat_dim, self.xyz_token_dim)
        # ★ 终极修改 1：将预测目标直接变为 3 维物理坐标 (X, Y, Z)
        # self.xyz_out_proj = nn.Linear(self.rgb_feat_dim, 3)
        
        # ==========================================
        # ★ 完美架构：双头预测 (Dual-Head)
        # 1. 语义头：负责重构 1152 维 PointMAE 特征，抓划痕
        self.xyz_feat_head = nn.Linear(self.rgb_feat_dim, self.xyz_token_dim)
        # 2. 坐标头：负责重构 3 维绝对物理坐标，抓形变
        self.xyz_coord_head = nn.Linear(self.rgb_feat_dim, 3)
        # ==========================================
        
        # Step 2.2: 潜在空间随机掩码
        # 维度要和你 PointMAE 投影后的维度一致 (768)
        self.masking_module = LatentRandomMasking(
            input_dim=self.rgb_feat_dim, # 768
            mask_ratio=0.75               # 文档要求 60%
        )

        self.spectral_transform = SpectralTransform()
        self.cbdg = DynamicContentGating(num_points=num_group)
        
        # ==========================================
        # ★ Step 3.3: 谱域自适应混合专家 (FD-MoE)
        # ==========================================
        # 配置: 768维, 4个专家, 每次选2个
        self.fd_moe = FD_MoE(
            channels=self.rgb_feat_dim, # 768
            num_experts=4, 
            top_k=2
        )

        # ★ Step 3.4 & 3.5: GGRM
        self.ggrm = GGRM(
            channels=self.rgb_feat_dim, # 768
            num_points=1024,            # M
            reduction=4                 # 通道缩放比
        )
        
        # ★ Phase 4: 空间互补分支
        self.spatial_branch = SpatialComplementBranch(channels=self.rgb_feat_dim)
        
        # ★ Phase 5: 融合模块
        self.fusion = DualStreamFusion(channels=self.rgb_feat_dim)
        
        # LapWaveGate：3D 几何高频增强
        # self.lapwave = LapWaveGate(
        #     in_channels=3,          # xyz 坐标
        #     k=16,
        #     lambda_edge=0.1,
        #     unet_out_channels=128,  # 输出 Cg
        #     alpha=1.0,
        #     gamma=0.6,
        #     s_list=(0.25, 0.5, 1.0, 2.0),
        #     K=8,
        #     alpha_w=1.2,
        #     gate_a=8.0,
        #     gate_tau_init=0.5,
        #     beta=0.4,
        # )

        # # LapWave 输出通道到 Point_MAE 输入的映射
        # self.lap2xyz = nn.Linear(128, 3)   # Cg=128 -> 3 维坐标增量 Δxyz

        # # LapWaveGate 的损失权重（等会在 ReconFeatures 里用）
        # self.lap_loss_weight = 1e-4

        # -------------------------
        # 双向 Cross-Attention：2D ↔ 3D
        # -------------------------
        self.bdca = BiDirectionalCrossAttention(
            dim_2d=self.rgb_feat_dim,     # 2D token 维度
            dim_3d=self.rgb_feat_dim,     # 3D token 维度（投影后）
            num_heads=8,
            drop=0.1,
            order="2d_first",
        )

        # -------------------------
        # Decoder：从融合后的特征重建 RGB 和 depth
        # -------------------------
        # RGB 解码：从 (B, C, 28, 28) -> (B, 3, 224, 224)
        # self.rgb_decoder = nn.Sequential(
        #     nn.Conv2d(self.rgb_feat_dim, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 28 -> 112
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 112 -> 224
        #     nn.Conv2d(128, 3, kernel_size=3, padding=1),
        #     nn.Sigmoid(),
        # )

        # depth 解码：从 (B, C, 8, 8) -> (B, 1, 224, 224)
        # 这里假定 num_group=64，可 reshape 为 8x8 的粗略“空间”
        # self.xyz_decoder = nn.Sequential(
        #     nn.Conv2d(self.rgb_feat_dim, 128, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  # 8 -> 32
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=7, mode="bilinear", align_corners=False),  # 32 -> 224
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1),
        # )
        # 4. 新的重建网络 (根据文档)
        # self.recon_2d = ReconNet2D(in_channels=self.rgb_feat_dim)
        # 计算特征图尺寸: 224 / 8 = 28
        feat_size = img_size // 8 
        
        self.recon_2d = ReconNet2D(
            in_channels=self.rgb_feat_dim, # 768
            embed_dim=96,                  # 内部通道数，Swin-Tiny标准是96，可调整
            img_size=feat_size,            # 显式传入 28
            window_size=7                  # 28 能被 7 整除，没问题
        )
        self.recon_3d = ReconNet3D(in_dim=self.rgb_feat_dim)
        
        
        # ★ 新增：用于永久存储训练集正常样本的像素级“误差基线”
        self.register_buffer('err_2d_mean', torch.zeros(1, img_size, img_size))
        self.register_buffer('err_2d_std', torch.ones(1, img_size, img_size))
        self.register_buffer('err_3d_mean', torch.zeros(1, img_size, img_size))
        self.register_buffer('err_3d_std', torch.ones(1, img_size, img_size))
        
        # ★ 新增：用于存储物理深度图的绝对基线
        self.register_buffer('depth_mean', torch.zeros(1, 1, img_size, img_size))
        self.register_buffer('depth_std', torch.ones(1, 1, img_size, img_size))
        
        # ==========================================
        # ★ 终极修复：必须冻结 Backbone 防止特征坍缩！
        # ==========================================
        for param in self.encoder.rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.encoder.xyz_backbone.parameters():
            param.requires_grad = False
            
        # 强制它们处于 eval 模式，防止 BatchNorm/Dropout 干扰提取出的纯净特征
        self.encoder.rgb_backbone.eval()
        self.encoder.xyz_backbone.eval()

    def forward(self, rgb, xyz,depth_map, ps):
        """
        rgb: (B, 3, 224, 224)
        ps : (B, 3, 224, 224)
        xyz: 
        - (B, 3, N)  组织点云被拉平成一维，N=H*W=224*224
        - 或 (B, 3, H, W) 原始网格
        """
        
        # 确保每次前向传播 Backbone 都在 eval 模式
        self.encoder.rgb_backbone.eval()
        self.encoder.xyz_backbone.eval()
        
        B = rgb.size(0)

        # -------------------------
        # 1) 2D & ps 特征提取（ViT）
        # -------------------------
        rgb_feats = self.encoder.forward_rgb_features(rgb)  # (B, 768, 28, 28)
        ps_feats  = self.encoder.forward_rgb_features(ps)   # (B, 768, 28, 28)

        # -------------------------
        # 2) CSS：2D & ps 之间做通道+空间交换
        # -------------------------
        rgb_feats_css, ps_feats_css = self.css(rgb_feats, ps_feats)
        # 后面重建 & cross-attn 用 rgb_feats_css，ps_feats_css 留给后续模块也可以
       
       # ==========================================
        # ★ 终极底牌：去噪自编码器 (Denoising Autoencoder)
        # 注入高斯噪声！既破坏了网络的“抄答案”能力，又保留了完整的空间锚点供 3D 参考。
        # 网络必须学会从噪声中“猜”出正常的零件，遇到缺陷时根本猜不出，从而爆红！
        # ==========================================
        # if self.training:
        #     # 注入标准差为 0.15 的高斯扰动
        #     noise = torch.randn_like(rgb_feats_css) * 0.15
        #     rgb_feats_bottleneck = rgb_feats_css + noise
        # else:
        #     rgb_feats_bottleneck = rgb_feats_css
        rgb_feats_bottleneck = rgb_feats_css.detach()
       
       
       # 兼容 xyz 输入格式
        if xyz.dim() == 4:
             # (B, 3, H, W) -> (B, N, 3)
            B, C, H, W = xyz.shape
            xyz = xyz.view(B, C, -1).transpose(1, 2).contiguous()


        # ==========================================
        # ★ 终极点云唤醒术：FPS 坐标坍缩 + 单位球归一化
        # ==========================================
        # 1. 获取前景掩码 (用你之前的 batch_min_z 方法)
        depth_for_mask = ps.mean(dim=1, keepdim=True) # 这里请换成你实际获取深度图的代码
        batch_min_z = depth_for_mask.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        fg_mask_224 = (depth_for_mask > batch_min_z + 1e-5).float()
        fg_mask_flat = fg_mask_224.view(B, -1, 1) # [B, 50176, 1]
        
        # 2. 坐标坍缩：背景坐标全变为 0，强迫 FPS 100% 采前景
        xyz_collapsed = xyz * fg_mask_flat

        # 3. 计算前景工件的物理质心 (Center of Mass)
        # 加 1e-5 防止除以 0
        valid_count = fg_mask_flat.sum(dim=1, keepdim=True) + 1e-5 
        center_of_mass = xyz_collapsed.sum(dim=1, keepdim=True) / valid_count # [B, 1, 3]
        
        # 4. 去中心化：把工件移动到 (0,0,0) 原点
        xyz_centered = (xyz_collapsed - center_of_mass) * fg_mask_flat
        
        # 5. 极值缩放：强制压缩到 [-1, 1] 的单位球内
        max_dist = torch.norm(xyz_centered, dim=-1, keepdim=True).max(dim=1, keepdim=True)[0] + 1e-5
        xyz_normalized = (xyz_centered / max_dist) * fg_mask_flat
            
            
        # -------------------------
        # 4) 3D 编码：Point_MAE
        # -------------------------
        # print("xyz.shape:",xyz.shape)
        xyz_tokens, center, ori_idx, center_idx = self.encoder.xyz_backbone(xyz_normalized)
        # ==========================================
        # ★ 新增: Phase 1 (Step 1.2 - 1.4)
        # 输入: center (采样后的几何坐标)
        # 输出: U (基底), loss_geo (几何正则化损失)
        # ==========================================
        U, loss_geo, knn_idx = self.shared_basis(center, ps, center_idx)
        
        # print("U.shape:",U.shape)
        
        # xyz_tokens: (B, 1152, G)  G = num_group = 64
        xyz_tokens = xyz_tokens.transpose(1, 2).contiguous()          # (B, G, 1152)
        xyz_tokens = torch.nan_to_num(xyz_tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        # ==========================================
        # ★ 终极修复 2：激活 3D 几何推断（摧毁死区）
        # 直接拿 PointMAE 提取的、富含绝对空间形变信息的 1152 维特征作为真值 (F_gt)！
        # ==========================================
        # F_gt = xyz_tokens.detach() 
        # ==========================================
        # ★ 终极修改 2：彻底抛弃语义特征，直接拿物理采样点 center (B, 1024, 3) 作为绝对真值！
        # ==========================================
        # F_gt = center.detach()
        
        # ★ 提取双真值 (都 detach 切断梯度)
        F_feat_gt = xyz_tokens.detach()   # 1152 维语义真值
        F_coord_gt = center.detach()      # 3 维坐标真值
        # print("Target STD:", F_feat_gt.std(dim=1).mean().item())
        # print("F_feat_gt.shape:",F_feat_gt.shape)
        
        # 投影到 768 维度，作为网络内部处理（Mask 和 频域变换）的起点
        F_in_raw = self.xyz_proj(xyz_tokens.detach())

        # ==========================================
        # ★ Step 2.2: 潜在空间随机掩码
        # 输入: F_gt
        # 输出: F_in (masked), mask (binary)
        # ==========================================
        if self.training:
            # 训练时：执行掩码
            F_in, mask = self.masking_module(F_in_raw)
        else:
            # 测试/推理时：通常不掩码，或者是为了做修复任务而掩码？
            # 工业异常检测的常见做法是：推理时也掩码，看模型能不能修补回来。
            # 如果你的逻辑是“重构误差”，那么测试时也需要掩码。
            # 根据 PointMAE 原理，测试时通常也进行掩码重建。
            F_in, mask = self.masking_module(F_in_raw)
            
            # 如果你想测试时全量输入(不掩码)，可以用:
            # F_in = F_in_raw
            # mask = torch.zeros(B, F_in_raw.shape[1], device=self.device)
        
        # ==========================================
        # ★ Phase 3 & 4 接口准备
        # ==========================================
        # 现在的 F_in 就是文档中 Step 3.1 的输入
        # F_in shape: [B, 1024, 768] (包含 [MASK] token)
        # U: [B, 1024, 1024], F_in: [B, 1024, 768]
        F_spec = self.spectral_transform.gft(F_in, U)  # -> [B, 1024, 768]
        # F_low, F_high 都是 [B, 1024, 768]
        # 但 F_low 主要保留了前段频率，F_high 保留了后段
        F_low, F_high, gate_map = self.cbdg(F_spec)
        
        # ==========================================
        # ★ Step 3.3: FD-MoE 处理低频分支
        # ==========================================
        # 输入: F_low (频域特征), U (用于内部逆变换)
        # 输出: F_low_spatial (修复后的完美几何特征，已变回空间域)
        F_low_spatial = self.fd_moe(F_low, U)
        
        # ==========================================
        # ★ Step 3.4 & 3.5: GGRM 处理高频并重组
        # ==========================================
        # 输入:
        # 1. Guide: F_low_spatial (空间域)
        # 2. Target: F_high (谱域, GGRM 内部会做逆变换)
        # 3. U: 正交基底
        xyz_features_processed = self.ggrm(F_low_spatial, F_high, U)
        
        # ==========================================
        # ★ Phase 4: 空间互补分支
        # 输入: F_in (掩码后的特征), knn_idx (完全相同的几何拓扑)
        # ==========================================
        F_spatial_out = self.spatial_branch(F_in, knn_idx)
        
        # ==========================================
        # ★ Phase 5: 融合与重建
        # ==========================================
        # 输入:
        # 1. 频域特征 (F_freq_out)
        # 2. 空间特征 (F_spatial_out)
        # 3. 原始输入 (F_in, 含 Mask Token, 用于残差连接)

        xyz_recon_final = self.fusion(
            f_freq_out=xyz_features_processed, 
            f_spatial_out=F_spatial_out, 
            f_in=F_in
        )
        
        # -------------------------
        # 5) 2D 特征 flatten 成 token 序列
        # -------------------------
        # B2, C_rgb, H_rgb, W_rgb = rgb_feats_css.shape             # C_rgb = 768, H=W=28
        # ★ 注意：这里使用的是带瓶颈的特征 rgb_feats_bottleneck！
        B2, C_rgb, H_rgb, W_rgb = rgb_feats_bottleneck.shape
        N_rgb = H_rgb * W_rgb

        # (B, 768, 28, 28) -> (B, 784, 768)
        # rgb_tokens = rgb_feats_css.view(B2, C_rgb, N_rgb).permute(0, 2, 1)
        rgb_tokens = rgb_feats_bottleneck.view(B2, C_rgb, N_rgb).permute(0, 2, 1)

        # -------------------------
        # 6) 双向 Cross-Attention：2D ↔ 3D
        # -------------------------
        rgb_updated, xyz_updated = self.bdca(rgb_tokens, xyz_recon_final)
        # ★ 黄金验证：强行切断 2D 给 3D 递小抄的通道！
        # rgb_updated = rgb_tokens
        # xyz_updated = xyz_recon_final
        # rgb_updated: (B, 784, 768)
        # xyz_updated: (B, G,  768)
        
        # -------------------------
        # 8) 解码重建
        # -------------------------
        # 2D 重建
        # Reshape back: [B, 768, 28, 28]
        # rgb_in_recon = rgb_updated.permute(0, 2, 1).view(B, C, H, W)
        B, N, D = rgb_updated.shape          # (B, 784, 768)
        h = w = int(N ** 0.5)               # 28
        assert h * w == N, (N, h, w)
        rgb_in_recon = rgb_updated.permute(0, 2, 1).contiguous().reshape(B, D, h, w)  # (B,768,28,28)
        # Output: [B, 768, 28, 28] -> Frgb (reconstructed)
        rgb_recon_feat = self.recon_2d(rgb_in_recon)
        # 3D 重建
        # Input: [B, G, 768] -> Output: [B, G, 768] -> Fpred
        xyz_recon_feat = self.recon_3d(xyz_updated,center)
        
        # ==========================================
        # ★ 终极修复 3：映射回 1152 维去计算损失
        # ==========================================
        # xyz_recon_final_out = self.xyz_out_proj(xyz_recon_feat)
        
        # ★ 分别通过两个头输出
        xyz_recon_feat_out = self.xyz_feat_head(xyz_recon_feat)   # [B, 1024, 1152]
        xyz_recon_coord_out = self.xyz_coord_head(xyz_recon_feat) # [B, 1024, 3]
        
        
        # # ==================== 泄漏探针开始 ====================
        # print("\n--- 3D 泄漏排查探针 ---")
        # # 1. 检查 Mask 极性是否反转
        # # 假设 N=1024, Mask_ratio=0.75。如果打印出 ~256，说明你算 Loss 算在了【可见点】上！可见点当然误差为0！
        # print(f"[探针 1] Mask 点数 (应该是 ~{int(xyz_tokens.shape[1] * 0.75)}):", mask[0].sum().item())
        
        # # 2. 检查 Masking 模块是否失效
        # # 如果 F_in 和 F_in_raw 是一样的，说明你的 masking_module 根本没有把特征替换成 Mask Token！
        # print("[探针 2] F_in 是否完全等于未掩码的 F_in_raw? :", torch.allclose(F_in, F_in_raw, atol=1e-6))
        
        # # 3. 检查 Mask Token 的纯度
        # # 真正的 Mask 区域，特征应该全部被替换成了同一个可学习的 mask_token，所以它的空间方差应该是 0.0！
        # # 如果打印出来不是 0.0，说明掩码区域依然带着原始信息！
        # mask_bool = mask[0].bool() # 拿第一个 batch 的 mask
        # if mask_bool.sum() > 0:
        #     print("[探针 3] Mask 区域内部的特征方差 (必须是 0.0!):", F_in[0, mask_bool, :].std(dim=0).mean().item())
            
        # # 4. 检查预测结果是不是直接复制了原始特征
        # # 如果网络什么都没做，直接把 F_in_raw 传过来了，这里的相似度会极高
        # print("[探针 4] 预测特征与 F_in_raw 的绝对差异:", (xyz_recon_feat - F_in_raw).abs().mean().item())
        # print("-----------------------\n")
        # # ==================== 泄漏探针结束 ====================
        
        return rgb_recon_feat, xyz_recon_feat_out, xyz_recon_coord_out, rgb_feats_css, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx


def splat_3d_error_to_2d_exact(errors, center_idx, img_size):
    B, N = errors.shape
    
    # 1. 直接在 224x224 的绝对分辨率画布上初始化
    error_map = torch.zeros((B, img_size * img_size), device=errors.device)
    
    # 过滤掉无效点 (center_idx == 0 的 padding 区域)
    valid_mask = (center_idx > 0).float()
    errors_clean = errors * valid_mask
    
    # 2. ★ 核心突围：直接映射绝对位置，坚决不求平均！
    # 将稀疏的误差峰值像钉钉子一样原封不动地砸在 224x224 画布上
    error_map.scatter_(1, center_idx.long(), errors_clean)
    error_map = error_map.view(B, 1, img_size, img_size)
    
    # 3. ★ 峰值膨胀连通 (Dilation)
    # 因为 1024 个点在 224x224 上是非常稀疏的星空状，
    # 使用 MaxPool 可以把那 1-2 个极亮的高分点直接扩张成一个红色的区块，而不被周围的 0 值稀释。
    # kernel_size=11 确保 1024 个点扩张后能无缝覆盖整个工件表面
    error_map_dilated = F.max_pool2d(error_map, kernel_size=11, stride=1, padding=5)
    
    # 4. 轻柔化边缘，消除方块马赛克感
    error_map_smooth = F.avg_pool2d(error_map_dilated, kernel_size=5, stride=1, padding=2)
    
    return error_map_smooth.squeeze(1)

# ------------------------------------------------
# 3. 重建式异常检测封装
# ------------------------------------------------
class ReconFeatures(nn.Module):
    """
    重建式异常检测：
      - train_step: 只算重建 loss (RGB + depth)
      - predict   : 输出融合重建误差 heatmap
    """

    def __init__(self, args):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.img_size_val = getattr(args, 'img_size', 224)
        self.net = FusionReconNet(
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group,
            img_size=args.img_size_val,
        ).to(self.device)

        self.blur = KNNGaussianBlur(2)
        
        # 超参数
        # l2_weight: 辅助损失的权重，通常设小一点 (e.g. 0.1 或 0.5)
        self.l2_weight = 0.5
        self.lambda_geo = getattr(args, 'lambda_geo', 0.1) # 从 args 读取，默认 0.1
        
        # ★ OT 模块 (用于训练 Loss 平衡)
        self.ot_module = UncertaintyAwareOT(momentum=0.9, init_temperature=0.2).to(self.device)
        
        # 指标与可视化缓存（仿 RealIAD-D3 的 Method 风格）
        self.reset_buffers()
        
        
        # ★ 终极调参中心：推理时控制三个模态得分上限的平衡系数
        self.weight_2d = 1.0           # 2D 基础权重
        self.weight_3d_feat = 2.0      # 3D 语义压制系数
        self.weight_depth = 0.3        # 物理深度压制系数 (因为方差极小，极易爆炸，必须重度压制)

    def reset_buffers(self):
        #   """清空评估阶段用到的缓存（仿 RealIAD-D3 的 Method）"""
        # 图像级
        self.image_preds = []   # 每张图的异常分数（如 max fmap）
        self.image_labels = []  # 每张图的标签 0/1

        # 像素级 (注意内存，如果显存/内存不足，不要存 pixel_preds)
        self.pixel_preds = []   # 所有样本的像素级异常分数
        self.pixel_labels = []  # 所有样本的像素级 GT (0/1)

        # 方便算 AU-PRO：每张图的 GT 和 Pred map
        self.gts = []           # list[np.ndarray]，每个 [H,W]
        self.pred_maps = []     # list[torch.Tensor] or np.ndarray，[H,W]

        # 可视化需要的原始输入（rgb, depth, ps）
        self.vis_samples = []   # list[(rgb_b, depth_b, ps_b)]
        
        
        # ★ 新增：初始化用于缓存可视化热力图的空列表
        self.maps_2d = [] 
        self.maps_3d = []

    def compute_hybrid_loss(self, pred, target, dim=-1, mask=None):
        """
        混合损失函数：强制 L2 归一化防坍缩
        """
        pred_safe = pred + 1e-8
        target_safe = target + 1e-8
        
        # 1. Cosine Loss
        cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim)
        
        # 2. L2 Loss (★ 核心修复：必须在 L2 归一化后计算，防止模型靠缩短向量长度作弊)
        pred_norm = F.normalize(pred, p=2, dim=dim)
        target_norm = F.normalize(target, p=2, dim=dim)
        
        l2_loss_raw = F.mse_loss(pred_norm, target_norm, reduction='none')
        
        if dim == 1: # 2D: [B, H, W]
            l2_loss = l2_loss_raw.mean(dim=1)
        else:        # 3D: [B, N]
            l2_loss = l2_loss_raw.mean(dim=-1)
            
        total_loss_map = cosine_loss + self.l2_weight * l2_loss
        
        # 3. 掩码过滤
        if mask is not None:
            total_loss_map = total_loss_map * mask
            return total_loss_map.sum() / (mask.sum() + 1e-5)
        else:
            return total_loss_map.mean()

    # -----------------------
    # 训练一步
    # -----------------------
    def train_step(self, sample):
        """
        sample: (rgb, xyz, depth_map, ps)
          - rgb        : (B, 3, 224, 224)
          - xyz        : (B, 3, N)
          - depth_map  : (B, 3, 224, 224)  (RealIAD 里的 z-channel 已经复制成3通道)
          - ps         : (B, 3, 224, 224)
        """
        rgb, xyz, depth_map, ps = sample

        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        ps = ps.to(self.device)

        # Forward
        # rgb_recon: 重建后的2D特征 [B, 768, 28, 28]
        # xyz_recon: 重建后的3D特征 [B, 1024, 768]
        # rgb_target: CSS后的2D特征 [B, 768, 28, 28]
        # xyz_target: 投影后的3D真值特征 [B, 1024, 768]
        # 注意这里精准解包了所有的 11 个返回值
        rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx = self.net(rgb, xyz, depth_map, ps)

        # ==========================================
        # ★ 核心大招：生成训练专用的 2D 损失掩码 (Loss Masking)
        # ==========================================
        depth_for_mask = depth_map.to(self.device)
        if depth_for_mask.dim() == 4:
            if depth_for_mask.shape[1] == 3: depth_for_mask = depth_for_mask[:, 2, :, :]
            elif depth_for_mask.shape[-1] == 3: depth_for_mask = depth_for_mask[:, :, :, 2]
            else: depth_for_mask = depth_for_mask.mean(dim=1)
        if depth_for_mask.dim() == 3:
            depth_for_mask = depth_for_mask.unsqueeze(1)
            
        batch_min_z = depth_for_mask.view(rgb.size(0), -1).min(dim=1)[0].view(rgb.size(0), 1, 1, 1)
        fg_mask_224 = (depth_for_mask > batch_min_z + 1e-5).float()
        
        # 向内腐蚀边缘 (kernel=15)，告诉网络：绝对不要去重建边缘！
        bg_mask = 1.0 - fg_mask_224
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=15, stride=1, padding=7)
        train_mask_224 = 1.0 - eroded_bg
        
        # 下采样到 28x28，适配 2D 特征图的大小
        train_mask_28 = F.interpolate(train_mask_224, size=(28, 28), mode='nearest').squeeze(1) # [B, 28, 28]

        # -------------------------
        # 计算 Loss
        # -------------------------
        
        # 1. 2D Loss: MSE(Frgb, Frgb')
        # 文档: "2d分支的损失函数是 Frgb 和 Frgb, 之间的误差"
        # 2D (Channel维度是 dim=1)
        # loss_2d = self.compute_hybrid_loss(rgb_recon, rgb_target.detach(),dim = 1) 
        loss_2d = self.compute_hybrid_loss(rgb_recon, rgb_target.detach(),dim = 1, mask=train_mask_28) 
        # 注意: 通常 Target 不需要梯度，detach 掉 rgb_target 以防止梯度传回 Encoder
        
        # 2. 3D Loss: Global MSE Loss
        # 文档: "Lrec = |Fpred - Fgt|^2"
        # 3D (Channel维度是 dim=2)
        # loss_3d = self.compute_hybrid_loss(xyz_recon, F_gt.detach(),dim = 2)
        # loss_3d = self.compute_hybrid_loss(xyz_recon, F_gt.detach(),dim = 2, mask=mask)
        # ==========================================
        # ★ 终极修改 3：3D Loss 变为纯粹的物理距离误差 (欧氏距离平方)
        # ==========================================
        # l2_loss_3d_raw = F.mse_loss(xyz_recon, F_gt.detach(), reduction='none') # [B, 1024, 3]
        # l2_loss_3d = l2_loss_3d_raw.mean(dim=-1) # [B, 1024]
        # loss_3d = (l2_loss_3d * mask).sum() / (mask.sum() + 1e-5)
        
        # 2. 3D 语义 Loss (带 F.normalize，防止 0.003 坍缩)
        # ==========================================
        # ★ 终极修复 1：提取 3D 点云的前景掩码
        # 将 2D 掩码展平，利用 center_idx 找出 1024 个点中哪些属于工件！
        # ==========================================
        B = rgb.size(0)
        fg_mask_flat = fg_mask_224.view(B, -1) # [B, 50176]
        fg_mask_3d = torch.gather(fg_mask_flat, 1, center_idx.long()) # [B, 1024]
        
        # 真正的有效掩码 = 被随机 Mask 遮挡的区域 AND 必须是工件前景！
        effective_3d_mask = mask * fg_mask_3d
        
        
        
        # 用有效掩码计算 3D Loss
        loss_3d_feat = self.compute_hybrid_loss(xyz_feat_recon, F_feat_gt, dim=2, mask=effective_3d_mask)
        
        # ==========================================
        # ★ 终极杀手锏：纯 Z 轴物理坐标监督 (完全抛弃XY死网格)
        # ==========================================
        pred_z = xyz_recon_coord_out[:, :, 2]   # [B, 1024]
        gt_z = F_coord_gt[:, :, 2].detach()     # [B, 1024]
        
        loss_3d_coord_raw = F.l1_loss(pred_z, gt_z, reduction='none') 
        loss_3d_coord = (loss_3d_coord_raw * effective_3d_mask).sum() / (effective_3d_mask.sum() + 1e-5)
        
        # 50.0 权重给纯粹的 Z 轴误差提权
        loss_3d = loss_3d_feat + 10.0 * loss_3d_coord
        # loss_3d = loss_3d_feat
        # loss_3d_feat = self.compute_hybrid_loss(xyz_feat_recon, F_feat_gt, dim=2, mask=mask)
        # loss_3d = loss_3d_feat 
        
        # # 3. ★ OT 计算权重 (传入 detach 防止作弊)
        # # 只有在训练时，我们希望"避重就轻"，让模型先学容易的，稳步收敛
        # alpha, beta = self.ot_module(loss_2d.detach(), loss_3d.detach())
        
        # # 取 Batch 平均权重进行反向传播
        # w_alpha = alpha.mean()
        # w_beta = beta.mean()
        
        # ---------- 换用绝对稳定的固定权重 ----------
        # 因为 2D Loss 稍微大一点点，1:1 或者 1:2 都是极佳的，这里推荐 1:1 稳如泰山
        w_alpha = 1.0
        w_beta = 1.0
    
        weighted_loss = w_alpha * loss_2d + w_beta * loss_3d
        total_loss = weighted_loss + self.lambda_geo * loss_geo
        
        return {
            "loss": total_loss,
            "l2d": loss_2d.item(),
            "l3d": loss_3d.item(),
            # ★ 新增：输出未乘系数的纯净版 3D 子 Loss
            "l3d_feat": loss_3d_feat.item(), 
            "l3d_coord": loss_3d_coord.item(),
            "geo": loss_geo.item(),
            "alpha": w_alpha,
            "beta": w_beta
        }
        
    @torch.no_grad()
    def build_error_statistics(self, train_loader):
        """
        在训练的最后一个 Epoch 结束后调用。
        遍历一次训练集，提取所有正常样本的平均重建误差和波动标准差，建立纯净基线。
        """
        self.net.eval()
        err_2d_list = []
        # err_3d_list = []
        err_3d_feat_list = []
        depth_list = []
        
        
        from tqdm import tqdm
        print("\n[Post-Training] Building Pixel-wise Z-Score Baseline...")
        for sample, _ in tqdm(train_loader, desc="Extracting Baseline"):
            rgb, xyz, depth_map, ps = sample
            rgb, xyz, ps = rgb.to(self.device), xyz.to(self.device), ps.to(self.device)
            B = rgb.size(0)

            rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, _, _, _, _, center_idx = self.net(rgb, xyz, depth_map, ps)
            # 1. 算 Raw 误差
            # === 彻底替换掉这 6 行旧代码 ===
            rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)
            rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
            # xyz_recon_norm = F.normalize(xyz_recon, p=2, dim=2)
            # F_gt_norm = F.normalize(F_gt, p=2, dim=2)

            # err_2d = torch.sum((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
            # if err_2d.shape[-1] != self.img_size_val:
            #     err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

            # err_3d_points = torch.sum((xyz_recon_norm - F_gt_norm) ** 2, dim=2)
            
            # === 彻底替换为真正的混合误差算子：余弦方向误差 + 绝对幅值误差 ===
            
            # 2D 误差图计算 (不归一化，保留绝对幅值)
            err_2d = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                     self.l2_weight * torch.mean((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
            
            if err_2d.shape[-1] != self.img_size_val:
                err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

            # ★ 3D 误差双重暴击
            # A. 语义误差：抓划痕
            xyz_feat_norm = F.normalize(xyz_feat_recon, p=2, dim=2)
            F_feat_gt_norm = F.normalize(F_feat_gt, p=2, dim=2)
            err_3d_feat = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                          self.l2_weight * torch.mean((xyz_feat_norm - F_feat_gt_norm) ** 2, dim=2)
            
            # ★ 加入纯 Z 轴基线计算
            pred_z = xyz_recon_coord_out[:, :, 2]
            gt_z = F_coord_gt[:, :, 2]
            err_3d_coord = torch.abs(pred_z - gt_z)
            
            total_3d_err = err_3d_feat + 10.0 * err_3d_coord
            err_3d_map = splat_3d_error_to_2d_exact(total_3d_err, center_idx, self.img_size_val)
            
            # err_3d_map = splat_3d_error_to_2d_exact(err_3d_feat, center_idx, self.img_size_val)

            # 3. 提取物理深度真值 (★ 为模板匹配建立基线)
            depth_gt = depth_map.to(self.device)
            if depth_gt.dim() == 4:
                if depth_gt.shape[1] == 3: depth_gt = depth_gt[:, 2:3, :, :]
                elif depth_gt.shape[-1] == 3: depth_gt = depth_gt[:, :, :, 2:3]
                else: depth_gt = depth_gt.mean(dim=1, keepdim=True)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)
                
            err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).squeeze(1)
            err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).squeeze(1)

            err_2d_list.append(err_2d_smooth)
            err_3d_feat_list.append(err_3d_smooth)
            depth_list.append(depth_gt.cpu())
            
        all_err_2d = torch.cat(err_2d_list, dim=0) # [Total_Samples, H, W]
        all_err_3d_feat = torch.cat(err_3d_feat_list, dim=0)
        all_depth = torch.cat(depth_list, dim=0)

        # 3. 注入模型的“记忆”中
        self.net.err_2d_mean.copy_(all_err_2d.mean(dim=0, keepdim=True))
        self.net.err_2d_std.copy_(all_err_2d.std(dim=0, keepdim=True) + 1e-5)
        self.net.err_3d_mean.copy_(all_err_3d_feat.mean(dim=0, keepdim=True))
        self.net.err_3d_std.copy_(all_err_3d_feat.std(dim=0, keepdim=True) + 1e-5)
        self.net.depth_mean.copy_(all_depth.mean(dim=0, keepdim=True))
        self.net.depth_std.copy_(all_depth.std(dim=0, keepdim=True) + 1e-5)
        print("Baseline and Physical Depth Template established and locked!")


    # -----------------------
    # 推理：累积式预测（仿 RealIAD-D3 的 Method）
    # -----------------------
    @torch.no_grad()
    def predict(self, sample, gt, label, rgb_path=None):
        if not isinstance(sample, (tuple, list)):
            raise RuntimeError(f"ReconFeatures.predict 期望 sample 是 tuple/list，实际类型: {type(sample)}")

        rgb, xyz, depth_map, ps = sample
        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        depth_map = depth_map.to(self.device)
        ps = ps.to(self.device)
        B = rgb.size(0)
        
        ensemble_runs = 4  
        err_3d_total_ensemble = 0

        for _ in range(ensemble_runs):
            rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, _, _, _, _, center_idx = self.net(rgb, xyz, depth_map, ps)
            
            xyz_feat_norm = F.normalize(xyz_feat_recon, p=2, dim=2)
            F_feat_gt_norm = F.normalize(F_feat_gt, p=2, dim=2)
            
            err_3d_feat_current = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                                  self.l2_weight * torch.mean((xyz_feat_norm - F_feat_gt_norm) ** 2, dim=2)
            
            # ★ 仅依赖 Z 轴下陷误差进行微小缺陷判定
            pred_z = xyz_recon_coord_out[:, :, 2]
            gt_z = F_coord_gt[:, :, 2]
            err_3d_coord_current = torch.abs(pred_z - gt_z)
            
            err_3d_total_ensemble += (err_3d_feat_current + 10.0 * err_3d_coord_current)

        err_3d_total = err_3d_total_ensemble / ensemble_runs

        rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)   
        rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
        
        err_2d = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                 self.l2_weight * torch.mean((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
                 
        if err_2d.shape[-1] != self.img_size_val:
            err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

        err_3d_map = splat_3d_error_to_2d_exact(err_3d_total, center_idx, self.img_size_val)

        depth_for_eval = depth_map.to(self.device)
        if depth_for_eval.dim() == 4:
            if depth_for_eval.shape[1] == 3: depth_for_eval = depth_for_eval[:, 2:3, :, :]
            elif depth_for_eval.shape[-1] == 3: depth_for_eval = depth_for_eval[:, :, :, 2:3]
            else: depth_for_eval = depth_for_eval.mean(dim=1, keepdim=True)
        if depth_for_eval.dim() == 3:
            depth_for_eval = depth_for_eval.unsqueeze(1)
            
        batch_min_z = depth_for_eval.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        fg_mask_raw = (depth_for_eval > batch_min_z + 1e-5).float()
        
        err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).to(self.device)
        err_2d_smooth = err_2d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).to(self.device)
        err_3d_smooth = err_3d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        std_2d = self.net.err_2d_std.to(self.device)
        std_3d_feat = self.net.err_3d_std.to(self.device)
        
        # ==========================================
        # ★ 极其关键的硬底线防爆：镇压底层浮点噪声放大的红斑！
        # ==========================================
        eps_2d = torch.clamp(std_2d.mean() * 0.1, min=1e-2)
        eps_3d = torch.clamp(std_3d_feat.mean() * 0.1, min=1e-2) 
        
        err_2d_norm = F.relu(err_2d_smooth - self.net.err_2d_mean.to(self.device)) / (std_2d + eps_2d)
        err_3d_feat_norm = F.relu(err_3d_smooth - self.net.err_3d_mean.to(self.device)) / (std_3d_feat + eps_3d)

        std_depth = self.net.depth_std.to(self.device)
        eps_depth = torch.clamp(std_depth.mean() * 0.1, min=1e-2)
        
        err_depth_raw = torch.abs(depth_for_eval - self.net.depth_mean.to(self.device))
        err_depth_norm = err_depth_raw.squeeze(1) / (std_depth.squeeze(1) + eps_depth)

        fg_mask_dilated = F.max_pool2d(fg_mask_raw, kernel_size=7, stride=1, padding=3)
        fg_mask_eval = F.interpolate(fg_mask_dilated, size=(self.img_size_val, self.img_size_val), mode='nearest').squeeze(1)
        
        err_2d_norm = err_2d_norm * fg_mask_eval
        err_3d_feat_norm = err_3d_feat_norm * fg_mask_eval
        err_depth_norm = err_depth_norm * fg_mask_eval

        fused = self.weight_2d * err_2d_norm + \
                self.weight_3d_feat * err_3d_feat_norm + \
                self.weight_depth * err_depth_norm

        bg_mask = 1.0 - fg_mask_raw
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=5, stride=1, padding=2)
        score_mask = (1.0 - eroded_bg).squeeze(1).cpu()
        
        gt = gt.detach().cpu()
        label = label.detach().cpu()
        fused = fused.cpu()
        for b in range(B):
            gmap = gt[b, 0]           
            fmap = fused[b]          

            fmap_score = fmap * score_mask[b]
            fmap_flat = fmap_score.flatten()
            k_val = max(1, int(fmap_flat.numel() * 0.01)) 
            topk_vals, _ = torch.topk(fmap_flat, k_val)
            img_score = float(topk_vals.mean())
            
            self.image_preds.append(img_score)
            self.image_labels.append(int(label[b]))

            self.gts.append(gmap.numpy())
            self.pred_maps.append(fmap.clone())
            
            self.maps_2d.append(err_2d_norm[b].clone())
            self.maps_3d.append(err_3d_feat_norm[b].clone())

            if isinstance(sample, (tuple, list)):
                rgb_b   = sample[0][b:b+1].cpu()
                depth_b = sample[2][b:b+1].cpu()
                ps_b    = sample[3][b:b+1].cpu()
            else:
                rgb_b = depth_b = ps_b = None

            self.vis_samples.append((rgb_b, depth_b, ps_b))

        return

    def calculate_metrics(self):
        from sklearn.metrics import roc_auc_score
        from utils.au_pro_util import calculate_au_pro

        if len(self.image_labels) == 0:
            self.image_rocauc = float("nan")
            self.pixel_rocauc = float("nan")
            self.au_pro = float("nan")
            return

        if len(set(self.image_labels)) > 1:
            self.image_rocauc = float(roc_auc_score(self.image_labels, self.image_preds))
        else:
            self.image_rocauc = float("nan")

        gts_flat = np.concatenate([gt.flatten() for gt in self.gts])
        preds_flat = np.concatenate([pred.cpu().numpy().flatten() for pred in self.pred_maps])
        
        assert gts_flat.shape == preds_flat.shape, f"Shape mismatch: GT {gts_flat.shape} vs Pred {preds_flat.shape}"
        self.pixel_rocauc = float(roc_auc_score(gts_flat, preds_flat))
