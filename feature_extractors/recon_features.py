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
        self.xyz_out_proj = nn.Linear(self.rgb_feat_dim, self.xyz_token_dim)
        
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

    def forward(self, rgb, xyz, ps):
        """
        rgb: (B, 3, 224, 224)
        ps : (B, 3, 224, 224)
        xyz: 
        - (B, 3, N)  组织点云被拉平成一维，N=H*W=224*224
        - 或 (B, 3, H, W) 原始网格
        """
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
        # -------------------------
        # 4) 3D 编码：Point_MAE
        # -------------------------
        xyz_tokens, center, ori_idx, center_idx = self.encoder.xyz_backbone(xyz)
        # ==========================================
        # ★ 新增: Phase 1 (Step 1.2 - 1.4)
        # 输入: center (采样后的几何坐标)
        # 输出: U (基底), loss_geo (几何正则化损失)
        # ==========================================
        U, loss_geo, knn_idx = self.shared_basis(center, ps, center_idx)
        
        # xyz_tokens: (B, 1152, G)  G = num_group = 64
        xyz_tokens = xyz_tokens.transpose(1, 2).contiguous()          # (B, G, 1152)
        xyz_tokens = torch.nan_to_num(xyz_tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        # ==========================================
        # ★ 终极修复 2：激活 3D 几何推断（摧毁死区）
        # 直接拿 PointMAE 提取的、富含绝对空间形变信息的 1152 维特征作为真值 (F_gt)！
        # ==========================================
        F_gt = xyz_tokens.detach() 
        
        # 投影到 768 维度，作为网络内部处理（Mask 和 频域变换）的起点
        F_in_raw = self.xyz_proj(xyz_tokens)

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
            # F_in = F_gt
            # mask = torch.zeros_like(F_gt[:,:,0])
        
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
        xyz_recon_feat = self.recon_3d(xyz_updated)
        
        # ==========================================
        # ★ 终极修复 3：映射回 1152 维去计算损失
        # ==========================================
        xyz_recon_final_out = self.xyz_out_proj(xyz_recon_feat)
        
        return rgb_recon_feat, xyz_recon_final_out, rgb_feats_css, F_gt, loss_geo, U, mask, center, center_idx


# ------------------------------------------------
# 辅助函数：将稀疏的 3D 误差投影回 2D 图像
# ------------------------------------------------
# def splat_3d_error_to_2d_exact(errors, center_idx, img_size):
#     """
#     errors: [B, N] 离散点的重建误差 (N=1024)
#     center_idx: [B, N] 这些点在原始 224x224 展平后的绝对像素索引
#     """
#     B, N = errors.shape
#     H = W = img_size
    
#     # 初始化展平的画布 [B, H*W]
#     error_map = torch.zeros((B, H * W), device=errors.device)
#     count_map = torch.zeros((B, H * W), device=errors.device)
    
#     # ★ 修复 2：将 center_idx == 0 的无效填充点的误差强制清零
#     # 这能防止所有无效点砸在左上角
#     valid_mask = (center_idx > 0).float()
#     errors_clean = errors * valid_mask
    
#     # ★ 绝对精确映射：直接用中心点原来的索引将误差填回对应的像素
#     error_map.scatter_add_(1, center_idx.long(), errors_clean)
#     count_map.scatter_add_(1, center_idx.long(), torch.ones_like(valid_mask))
    
#     # 计算平均值
#     mask = count_map > 0
#     error_map[mask] /= count_map[mask]
    
#     # 变回 2D 图像维度 [B, H, W]
#     error_map = error_map.view(B, H, W)
    
#     # 扩散填补 FPS 采样带来的空洞
#     # error_map = F.max_pool2d(error_map.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
#     # # ★ 修复 3D 马赛克：
#     # # 1. 用 kernel=11 的 MaxPool 充分扩张，确保 1024 个点能完全覆盖所有空洞
#     # error_map = F.max_pool2d(error_map.unsqueeze(1), kernel_size=11, stride=1, padding=5)
    
#     # # 2. 用 kernel=11 的 AvgPool 抹平方形的马赛克边缘，使其变成平滑的地形
#     # error_map = F.avg_pool2d(error_map, kernel_size=11, stride=1, padding=5).squeeze(1)
    
#     # ★ 修复 3D 偏移：废除 max_pool 的膨胀效应，改用大核 AvgPool 均匀扩散，确保重心绝对居中！
#     # error_map = F.avg_pool2d(error_map.unsqueeze(1), kernel_size=11, stride=1, padding=5)
#     # error_map = F.avg_pool2d(error_map, kernel_size=5, stride=1, padding=2).squeeze(1)
#     # === 替换为：先用紧凑的 MaxPool 填补采样空洞，再用小核 AvgPool 柔化边缘 ===
#     error_map = F.max_pool2d(error_map.unsqueeze(1), kernel_size=7, stride=1, padding=3)
#     # 取消掉 AvgPool，保持病灶边缘锋利
#     error_map = F.avg_pool2d(error_map, kernel_size=3, stride=1, padding=1).squeeze(1)
#     # error_map = error_map.squeeze(1)
    
    
#     return error_map

# def splat_3d_error_to_2d_exact(errors, center_idx, img_size):
#     B, N = errors.shape
    
#     # 1. 降采样映射到低分辨率特征网格 (28x28)
#     small_size = 28
#     scale = img_size / small_size  # 224 / 28 = 8.0
    
#     y = center_idx // img_size
#     x = center_idx % img_size
    
#     # 向下取整映射到 28x28 对应的索引
#     y_small = torch.clamp((y.float() / scale).long(), 0, small_size - 1)
#     x_small = torch.clamp((x.float() / scale).long(), 0, small_size - 1)
#     idx_small = y_small * small_size + x_small
    
#     error_map_small = torch.zeros((B, small_size * small_size), device=errors.device)
#     count_map_small = torch.zeros((B, small_size * small_size), device=errors.device)
    
#     valid_mask = (center_idx > 0).float()
#     errors_clean = errors * valid_mask
    
#     # 在 28x28 网格上累加误差
#     error_map_small.scatter_add_(1, idx_small.long(), errors_clean)
#     count_map_small.scatter_add_(1, idx_small.long(), torch.ones_like(valid_mask))
    
#     # 求平均
#     mask = count_map_small > 0
#     error_map_small[mask] /= count_map_small[mask]
#     error_map_small = error_map_small.view(B, 1, small_size, small_size)
    
#     # 2. 填补 28x28 网格中的微小缝隙（因为非常密集，只需极小的 kernel=3 即可）
#     error_map_small = F.max_pool2d(error_map_small, kernel_size=3, stride=1, padding=1)
    
#     # 3. 终极平滑：双线性插值放大回 224x224，告别马赛克方块！
#     error_map = F.interpolate(error_map_small, size=(img_size, img_size), mode='bilinear', align_corners=False)
    
#     return error_map.squeeze(1)
def splat_3d_error_to_2d_exact(errors, center_idx, img_size):
    B, N = errors.shape
    
    # 1. 降采样映射到低分辨率特征网格 (28x28)
    small_size = 28
    scale = img_size / small_size  # 224 / 28 = 8.0
    
    y = center_idx // img_size
    x = center_idx % img_size
    
    # 向下取整映射到 28x28 对应的索引
    y_small = torch.clamp((y.float() / scale).long(), 0, small_size - 1)
    x_small = torch.clamp((x.float() / scale).long(), 0, small_size - 1)
    idx_small = y_small * small_size + x_small
    
    error_map_small = torch.zeros((B, small_size * small_size), device=errors.device)
    count_map_small = torch.zeros((B, small_size * small_size), device=errors.device)
    
    valid_mask = (center_idx > 0).float()
    errors_clean = errors * valid_mask
    
    # 在 28x28 网格上累加误差
    error_map_small.scatter_add_(1, idx_small.long(), errors_clean)
    count_map_small.scatter_add_(1, idx_small.long(), torch.ones_like(valid_mask))
    
    # 求平均
    mask = count_map_small > 0
    error_map_small[mask] /= count_map_small[mask]
    
    error_map_small = error_map_small.view(B, 1, small_size, small_size)
    count_map_small = count_map_small.view(B, 1, small_size, small_size)
    
    # ==========================================
    # ★ 终极收缩：智能空洞填补 (Smart Hole-Filling)
    # 只对没有采到点的空洞(0值区)进行邻域借值，绝不向外膨胀原本的高亮缺陷！
    # ==========================================
    empty_mask = (count_map_small == 0).float()
    error_map_dilated = F.max_pool2d(error_map_small, kernel_size=3, stride=1, padding=1)
    
    # 有点的地方保持原始锐利度，只有空洞才会加上膨胀值
    error_map_small = error_map_small + error_map_dilated * empty_mask
    
    # 3. 终极平滑：双线性插值放大回 224x224
    error_map = F.interpolate(error_map_small, size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    return error_map.squeeze(1)

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

    # def compute_hybrid_loss(self, pred, target, dim=-1):
    #     """
    #     混合损失函数：Cosine (主) + L2 (辅)
    #     """
    #     # 1. Cosine Loss (主): 1 - cos_sim
    #     # pred, target: [B, N, C] or [B, C, H, W]
    #     # ★ 终极防线 3：强制加上极小值，防止纯黑图像/零特征引发 NaN
    #     pred_safe = pred + 1e-8
    #     target_safe = target + 1e-8
    #     cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim).mean()
        
    #     # 2. L2 Loss (辅): MSE
    #     l2_loss = F.mse_loss(pred, target)
        
    #     return cosine_loss + self.l2_weight * l2_loss
    
    # def compute_hybrid_loss(self, pred, target, dim=-1, mask=None):
    #     """
    #     混合损失函数：支持传入 mask，只计算掩码区域的 Loss
    #     """
    #     pred_safe = pred + 1e-8
    #     target_safe = target + 1e-8
        
    #     # 1. Cosine Loss
    #     # 2D: [B, H, W] | 3D: [B, N]
    #     cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim)
        
    #     # 2. L2 Loss
    #     # [B, C, H, W] 或 [B, N, C]
    #     l2_loss_raw = F.mse_loss(pred, target, reduction='none')
    #     if dim == 1: # 2D: 对通道求平均 -> [B, H, W]
    #         l2_loss = l2_loss_raw.mean(dim=1)
    #     else:        # 3D: 对通道求平均 -> [B, N]
    #         l2_loss = l2_loss_raw.mean(dim=-1)
            
    #     total_loss_map = cosine_loss + self.l2_weight * l2_loss
        
    #     # 3. 掩码过滤
    #     if mask is not None:
    #         total_loss_map = total_loss_map * mask
    #         # 只求有效区域的平均值
    #         return total_loss_map.sum() / (mask.sum() + 1e-5)
    #     else:
    #         return total_loss_map.mean()
    def compute_hybrid_loss(self, pred, target, dim=-1, mask=None):
        """
        混合损失函数：强制 L2 归一化防坍缩
        """
        pred_safe = pred + 1e-8
        target_safe = target + 1e-8
        
        # 1. Cosine Loss
        cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim)
        
        # 2. L2 Loss (★ 核心修复：必须在 L2 归一化后计算，防止模型靠缩短向量长度作弊)
        # pred_norm = F.normalize(pred, p=2, dim=dim)
        # target_norm = F.normalize(target, p=2, dim=dim)
        
        l2_loss_raw = F.mse_loss(pred, target, reduction='none')
        
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
        rgb_recon, xyz_recon, rgb_target, F_gt, loss_geo, U, mask, center, center_idx = self.net(rgb, xyz, ps)

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
        loss_3d = self.compute_hybrid_loss(xyz_recon, F_gt.detach(),dim = 2, mask=mask)
        
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
        
        # return {
        #     "loss": total_loss,
        #     "l2d": loss_2d.item(),
        #     "l3d": loss_3d.item(),
        #     "geo": loss_geo.item(),
        #     "alpha": w_alpha.item(),
        #     "beta": w_beta.item()
        # }
        return {
            "loss": total_loss,
            "l2d": loss_2d.item(),
            "l3d": loss_3d.item(),
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
        err_3d_list = []
        
        from tqdm import tqdm
        print("\n[Post-Training] Building Pixel-wise Z-Score Baseline...")
        for sample, _ in tqdm(train_loader, desc="Extracting Baseline"):
            rgb, xyz, depth_map, ps = sample
            rgb, xyz, ps = rgb.to(self.device), xyz.to(self.device), ps.to(self.device)
            B = rgb.size(0)

            rgb_recon, xyz_recon, rgb_target, F_gt, _, _, _, _, center_idx = self.net(rgb, xyz, ps)

            # 1. 算 Raw 误差
            # === 彻底替换掉这 6 行旧代码 ===
            # rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)
            # rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
            # xyz_recon_norm = F.normalize(xyz_recon, p=2, dim=2)
            # F_gt_norm = F.normalize(F_gt, p=2, dim=2)

            # err_2d = torch.sum((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
            # if err_2d.shape[-1] != self.img_size_val:
            #     err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

            # err_3d_points = torch.sum((xyz_recon_norm - F_gt_norm) ** 2, dim=2)
            
            # === 彻底替换为真正的混合误差算子：余弦方向误差 + 绝对幅值误差 ===
            
            # 2D 误差图计算 (不归一化，保留绝对幅值)
            err_2d = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                     self.l2_weight * torch.mean((rgb_recon - rgb_target) ** 2, dim=1)
            
            if err_2d.shape[-1] != self.img_size_val:
                err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

            # 3D 误差图计算 (不归一化，捕捉形变带来的空间偏移)
            err_3d_points = (1 - F.cosine_similarity(xyz_recon + 1e-8, F_gt + 1e-8, dim=2)) + \
                            self.l2_weight * torch.mean((xyz_recon - F_gt) ** 2, dim=2)
            # ★ 此处附带了你的 3D 偏移修复！
            err_3d_map = splat_3d_error_to_2d_exact(err_3d_points, center_idx, self.img_size_val)

            # 2. 收集平滑后的空间特征
            err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).squeeze(1)
            err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).squeeze(1)

            err_2d_list.append(err_2d_smooth)
            err_3d_list.append(err_3d_smooth)
            
        all_err_2d = torch.cat(err_2d_list, dim=0) # [Total_Samples, H, W]
        all_err_3d = torch.cat(err_3d_list, dim=0)

        # 3. 注入模型的“记忆”中
        self.net.err_2d_mean.copy_(all_err_2d.mean(dim=0, keepdim=True))
        self.net.err_2d_std.copy_(all_err_2d.std(dim=0, keepdim=True) + 1e-5)
        self.net.err_3d_mean.copy_(all_err_3d.mean(dim=0, keepdim=True))
        self.net.err_3d_std.copy_(all_err_3d.std(dim=0, keepdim=True) + 1e-5)
        print("Baseline established and locked into model registry!")


    # -----------------------
    # 推理：累积式预测（仿 RealIAD-D3 的 Method）
    # -----------------------
    @torch.no_grad()
    def predict(self, sample, gt, label, rgb_path=None):
        """
        基于特征重建误差进行预测
        推理步：直接融合 (Summation)
        原因：推理时的高误差通常代表缺陷，不能被 OT 抑制。
        """
        # -------------------------
        # 1. 解包 sample，兼容 3 / 4 元素
        # -------------------------
        if not isinstance(sample, (tuple, list)):
            raise RuntimeError(
                f"ReconFeatures.predict 期望 sample 是 tuple/list，实际类型: {type(sample)}"
            )

        rgb, xyz, depth_map, ps = sample

        # -------------------------
        # 2. 丢到 device 上
        # -------------------------
        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        depth_map = depth_map.to(self.device)
        ps = ps.to(self.device)
        B = rgb.size(0)

        # -------------------------
        # 3. 前向重建（注意要把 ps 也传进去）
        # -------------------------
        rgb_recon, xyz_recon, rgb_target, F_gt, _, _, _, center, center_idx = self.net(rgb, xyz, ps)
        # rgb_recon: [B, 3, H, W]
        # xyz_recon: [B, 1, H, W]
        # === 调试打印 ===
        # print(f"Center Range - X: [{center[..., 0].min().item():.2f}, {center[..., 0].max().item():.2f}]")
        # print(f"Center Range - Y: [{center[..., 1].min().item():.2f}, {center[..., 1].max().item():.2f}]")

        # -------------------------
        # 4. 重建误差 map
        # -------------------------
        # ---------------------------------------------------
        # 步骤 A: 计算误差数值 (L2 Norm + Euclidean)
        # ---------------------------------------------------
        
        # # 1. L2 归一化 (解决量级差异的关键)
        # rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)   # [B, C, H, W]
        # rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
        
        # xyz_recon_norm = F.normalize(xyz_recon, p=2, dim=2)   # [B, N, C]
        # F_gt_norm = F.normalize(F_gt, p=2, dim=2)

        # # 2. 计算欧氏距离平方 (逐像素/逐点)
        # # 2D Error: [B, H, W]
        # err_2d = torch.sum((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
        # if err_2d.shape[-1] != self.img_size_val:
        #     err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), 
        #                           mode='bilinear', align_corners=False).squeeze(1)
        
        # # ★ 修复 3：切除 2D 特征图因卷积/Swin带来的边缘平移伪影
        # # 强制将最外围一圈 (或两圈) 的误差清零
        # # err_2d[:, :, :2] = 0   # 左边缘
        # # err_2d[:, :, -2:] = 0  # 右边缘
        # # err_2d[:, :2, :] = 0   # 上边缘
        # # err_2d[:, -2:, :] = 0  # 下边缘
        
        # # 3D Error Points: [B, N] (这是 N 个离散点的误差值)
        # err_3d_points = torch.sum((xyz_recon_norm - F_gt_norm) ** 2, dim=2)
        
        # === 替换为 ===
        err_2d = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                 self.l2_weight * torch.mean((rgb_recon - rgb_target) ** 2, dim=1)
                 
        if err_2d.shape[-1] != self.img_size_val:
            err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)

        err_3d_points = (1 - F.cosine_similarity(xyz_recon + 1e-8, F_gt + 1e-8, dim=2)) + \
                        self.l2_weight * torch.mean((xyz_recon - F_gt) ** 2, dim=2)
        
        # ---------------------------------------------------
        # 步骤 B: 空间映射 (Splatting) - 这一步绝对不能省！
        # ---------------------------------------------------
        
        # 利用 center 坐标，将离散的 err_3d_points 投影回 2D 网格
        # 如果没有这一步，3D 误差图就是乱序的噪声
        # err_3d_map = splat_3d_error_to_2d(err_3d_points, center, self.img_size_val)
        # ★ 2. 替换旧的调用方法
        # 抛弃原来按坐标乱投的 err_3d_map = splat_3d_error_to_2d(err_3d_points, center, self.img_size_val)
        err_3d_map = splat_3d_error_to_2d_exact(err_3d_points, center_idx, self.img_size_val)

        # ---------------------------------------------------
        # 步骤 C: 后处理与融合
        # ---------------------------------------------------

        if depth_map.dim() == 4:
            if depth_map.shape[1] == 3: depth_map = depth_map[:, 2, :, :]
            elif depth_map.shape[-1] == 3: depth_map = depth_map[:, :, :, 2]
            else: depth_map = depth_map.mean(dim=1)
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
            
        batch_min_z = depth_map.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        # ★ 完整的物体 Mask (一刀不剪，捍卫 Pixel AUC！)
        fg_mask_raw = (depth_map > batch_min_z + 1e-5).float()
        
        # # ★ 新增：向内腐蚀前景掩码 3 个像素，彻底切掉最边缘的插值红光！
        # bg_mask_raw = 1.0 - fg_mask_raw
        # # eroded_bg_raw = F.max_pool2d(bg_mask_raw, kernel_size=7, stride=1, padding=3)
        # # 改为轻度腐蚀，仅去除 1 个像素的极边缘噪点：
        # eroded_bg_raw = F.max_pool2d(bg_mask_raw, kernel_size=3, stride=1, padding=1)
        # fg_mask_clean = 1.0 - eroded_bg_raw
        
        # fg_mask = F.interpolate(fg_mask_clean, size=(self.img_size_val, self.img_size_val), mode='nearest').squeeze(1)
        
        # 高斯模糊 (平滑噪声)
        # err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).squeeze(1).to(self.device)
        # err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).squeeze(1).to(self.device)
        # 4. 平滑 (增加维度强校验，防止 blur 吞掉 Batch 维度)
        err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).to(self.device)
        err_2d_smooth = err_2d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).to(self.device)
        err_3d_smooth = err_3d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        # # ==========================================
        # # ★ 终极逻辑：纯净均值相减 (Baseline Subtraction)
        # # 减去训练集均值，消灭金属误报。用 ReLU 砍掉负值，只保留真正溢出的缺陷！
        # # ==========================================
        # err_2d_norm = F.relu(err_2d_smooth - self.net.err_2d_mean.to(self.device))
        # err_3d_norm = F.relu(err_3d_smooth - self.net.err_3d_mean.to(self.device))
        
        # ================== D. ★ 终极奥义：鲁棒 Z-Score (Robust Z-Score) 异常放大器 ==================
        std_2d = self.net.err_2d_std.to(self.device)
        std_3d = self.net.err_3d_std.to(self.device)
        
        # ★ 修复：加入鲁棒保护垫 (Robust Epsilon)，彻底杜绝边缘误差爆炸！
        # 提取当前特征图标准差的全局均值的一个比例（如 10%），外加基础防爆常数 1e-3
        eps_2d = torch.clamp(std_2d.mean() * 0.1, min=1e-3)
        eps_3d = torch.clamp(std_3d.mean() * 0.1, min=1e-3)
        
        # 使用 (误差 - 均值) / (标准差 + 保护垫)
        err_2d_norm = F.relu(err_2d_smooth - self.net.err_2d_mean.to(self.device)) / (std_2d + eps_2d)
        err_3d_norm = F.relu(err_3d_smooth - self.net.err_3d_mean.to(self.device)) / (std_3d + eps_3d)

        # ==========================================
        # ★ 黄金 5 像素宽容评估掩码 (kernel=11, padding=5)
        # 拯救 Impact Damage 的边缘破损！
        # ==========================================
        fg_mask_dilated = F.max_pool2d(fg_mask_raw, kernel_size=11, stride=1, padding=5)
        fg_mask_eval = F.interpolate(fg_mask_dilated, size=(self.img_size_val, self.img_size_val), mode='nearest').squeeze(1)

        # 乘上前景掩码，确保背景绝对干净
        err_2d_norm = err_2d_norm * fg_mask_eval
        err_3d_norm = err_3d_norm * fg_mask_eval

        # 5. ★ 推理融合：直接相加 (等价于 alpha=0.5, beta=0.5)
        # 这样任何一个分支检测到的缺陷 (高误差) 都会被保留
        fused = err_2d_norm + err_3d_norm

        # ★ 6. 专门给 Image AUC 准备的【重度打分掩码】
        # 向内疯狂收缩 15 像素，确保 Image AUC 持续飙升！但不改变存下来的图！
        bg_mask = 1.0 - fg_mask_raw
        # eroded_bg = F.max_pool2d(bg_mask, kernel_size=11, stride=1, padding=5)
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=5, stride=1, padding=2)
        score_mask = (1.0 - eroded_bg).squeeze(1).cpu()
        
        # ★ 取消疯狂腐蚀的 score_mask，直接使用原版前景 Mask，让边缘的 Impact Damage 也能得分！
        # score_mask = fg_mask.cpu()

        # -------------------------
        # 7. 仿 RealIAD-D3：把 score 累积到 buffer 里
        # -------------------------
        gt = gt.detach().cpu()
        label = label.detach().cpu()
        fused = fused.cpu()
        for b in range(B):
            gmap = gt[b, 0]           # [Hg, Wg]
            fmap = fused[b]          # [1,H,W] / [H,W] / [1,L] / [L]

            # ========= 现在 fmap / gmap 都是 [Hg, Wg] =========
            # 算 Image AUC：套上重度掩码，取 max，保证 Image AUC 继续涨！
            fmap_score = fmap * score_mask[b]
            # ★ 修改前：img_score = float(fmap_score.max())
            # ★ 修改后：使用 Top-K 平滑聚合，防止边缘单一的高频噪点让正常样本得分暴涨
            fmap_flat = fmap_score.flatten()
            # 动态计算 1% 的像素数量，如果图比较小至少保留 1 个点
            k_val = max(1, int(fmap_flat.numel() * 0.01)) 
            topk_vals, _ = torch.topk(fmap_flat, k_val)
            img_score = float(topk_vals.mean())
            self.image_preds.append(img_score)
            self.image_labels.append(int(label[b]))

            # 存展平的像素 (慎用，内存大)
            # self.pixel_preds.extend(fmap.flatten().numpy().tolist())
            # self.pixel_labels.extend(gmap.flatten().numpy().tolist())

            self.gts.append(gmap.numpy())
            self.pred_maps.append(fmap.clone())
            
            # --- 保存单独的 Map ---
            self.maps_2d.append(err_2d_norm[b].clone())
            self.maps_3d.append(err_3d_norm[b].clone())

            # 为可视化缓存对应的 rgb / depth / ps
            if isinstance(sample, (tuple, list)):
                # (rgb, xyz, depth_map, ps)
                rgb_b   = sample[0][b:b+1].cpu()
                depth_b = sample[2][b:b+1].cpu()
                ps_b    = sample[3][b:b+1].cpu()
            else:
                rgb_b = depth_b = ps_b = None

            self.vis_samples.append((rgb_b, depth_b, ps_b))

        # 仿 RealIAD：predict 不返回值，只是累积缓存
        return


    def calculate_metrics(self):
        """
        仿 RealIAD-D3：根据累积的 buffer 计算 Image ROC / Pixel ROC / AU-PRO
        结果保存在：
          - self.image_rocauc
          - self.pixel_rocauc
          - self.au_pro
        """
        from sklearn.metrics import roc_auc_score
        from utils.au_pro_util import calculate_au_pro

        if len(self.image_labels) == 0:
            self.image_rocauc = float("nan")
            self.pixel_rocauc = float("nan")
            self.au_pro = float("nan")
            return

        # 图像级 ROC
        if len(set(self.image_labels)) > 1:
            self.image_rocauc = float(roc_auc_score(self.image_labels, self.image_preds))
        else:
            self.image_rocauc = float("nan")


        # 2. 计算像素级 (Pixel-level) AUROC
        # 利用推断阶段已经保存好的 gts (Mask) 和 pred_maps (热力图)
        # 将它们全部拉平 (flatten) 并拼接成两个超长的一维 numpy 数组
        
        # self.gts 里面是 numpy array，直接 flatten
        gts_flat = np.concatenate([gt.flatten() for gt in self.gts])
        
        # self.pred_maps 里面是 tensor，先转 numpy 再 flatten
        preds_flat = np.concatenate([pred.cpu().numpy().flatten() for pred in self.pred_maps])
        
        # 3. 确保两者维度完全一致后再送入 sklearn
        assert gts_flat.shape == preds_flat.shape, f"Shape mismatch: GT {gts_flat.shape} vs Pred {preds_flat.shape}"
        self.pixel_rocauc = float(roc_auc_score(gts_flat, preds_flat))
        # 像素级 ROC
        # self.pixel_rocauc = float(roc_auc_score(self.pixel_labels, self.pixel_preds))

        # -------- AU-PRO：确保传入 numpy 数组 --------
        # gts_np = [
        #     g if isinstance(g, np.ndarray) else g.detach().cpu().numpy()
        #     for g in self.gts
        # ]
        # preds_np = [
        #     p if isinstance(p, np.ndarray) else p.detach().cpu().numpy()
        #     for p in self.pred_maps
        # ]

        # au_pro, _ = calculate_au_pro(gts_np, preds_np)
        # self.au_pro = float(au_pro)
