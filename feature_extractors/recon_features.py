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
        
        # Step 2.2: 潜在空间随机掩码
        # 维度要和你 PointMAE 投影后的维度一致 (768)
        self.masking_module = LatentRandomMasking(
            input_dim=self.rgb_feat_dim, # 768
            mask_ratio=0.6               # 文档要求 60%
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
        # 投影到 768 维度 -> [B, G, 768]
        # 这就是文档中的 Fgt (Latent Ground Truth)
        F_gt = self.xyz_proj(xyz_tokens)

        # ==========================================
        # ★ Step 2.2: 潜在空间随机掩码
        # 输入: F_gt
        # 输出: F_in (masked), mask (binary)
        # ==========================================
        if self.training:
            # 训练时：执行掩码
            F_in, mask = self.masking_module(F_gt)
        else:
            # 测试/推理时：通常不掩码，或者是为了做修复任务而掩码？
            # 工业异常检测的常见做法是：推理时也掩码，看模型能不能修补回来。
            # 如果你的逻辑是“重构误差”，那么测试时也需要掩码。
            # 根据 PointMAE 原理，测试时通常也进行掩码重建。
            F_in, mask = self.masking_module(F_gt)
            
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
        B2, C_rgb, H_rgb, W_rgb = rgb_feats_css.shape             # C_rgb = 768, H=W=28
        N_rgb = H_rgb * W_rgb

        # (B, 768, 28, 28) -> (B, 784, 768)
        rgb_tokens = rgb_feats_css.view(B2, C_rgb, N_rgb).permute(0, 2, 1)

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
        
        return rgb_recon_feat, xyz_recon_feat, rgb_feats_css, F_gt, loss_geo, U, mask, center


# ------------------------------------------------
# 辅助函数：将稀疏的 3D 误差投影回 2D 图像
# ------------------------------------------------
def splat_3d_error_to_2d(errors, centers, img_size):
    """
    errors:  [B, N]  每个点的误差值
    centers: [B, N, 3] 每个点的坐标 (假设已归一化到 -0.x ~ 0.x 或类似范围)
    img_size: int (e.g. 224)
    """
    B, N = errors.shape
    H, W = img_size, img_size
    
    # 初始化 2D Map [B, 1, H, W]
    error_map = torch.zeros((B, 1, H, W), device=errors.device)
    # 计数器 (处理多个点落入同一个像素的情况)
    count_map = torch.zeros((B, 1, H, W), device=errors.device)
    
    # 1. 坐标归一化与映射
    # 假设 centers 的 x,y 范围通常在 [-0.5, 0.5] 或 [-1, 1] 之间
    # 这是一个比较粗暴的假设，如果你的点云坐标很大，这里需要先 normalize
    # 我们这里假设输入的一般是归一化后的点云
    
    # 获取 x, y (忽略 z)
    # 注意：点云坐标系和图像坐标系可能存在翻转，根据实际情况调整
    # 这里假设 x -> W (列), y -> H (行)
    
    # 1. 自适应归一化 (Per-sample Min-Max)
    # 不管你的 center 是 [0, 223] 还是 [-1, 1]，这里都会归一化到 [0, 1]
    min_c = centers.min(dim=1, keepdim=True)[0] # [B, 1, 3]
    max_c = centers.max(dim=1, keepdim=True)[0] # [B, 1, 3]
    range_c = max_c - min_c + 1e-6
    
    norm_centers = (centers - min_c) / range_c # [0, 1]
    
    # 2. 映射到像素坐标 (关键修改)
    # 根据你的 Dataset: Channel 0 是行(v), Channel 1 是列(u) 
    # Col (u) -> Channel 1
    u = (norm_centers[:, :, 1] * (W - 1)).long()
    # Row (v) -> Channel 0
    v = (norm_centers[:, :, 0] * (H - 1)).long()
    
    # 边界保护
    u = torch.clamp(u, 0, W - 1)
    v = torch.clamp(v, 0, H - 1)
    
    # 2. 填值 (Splatting)
    # 这种循环在 Python 里很慢，但在推理时 N=1024 还能接受
    # 更快的方法是使用 scatter_add，但需要处理索引展平
    for b in range(B):
        # 展平索引: v * W + u
        flat_indices = v[b] * W + u[b]
        
        # 展平 map
        flat_map = error_map[b].view(-1)
        flat_count = count_map[b].view(-1)
        
        # 累加误差
        flat_map.scatter_add_(0, flat_indices, errors[b])
        flat_count.scatter_add_(0, flat_indices, torch.ones_like(errors[b]))
        
        # 还原
        error_map[b] = flat_map.view(1, H, W)
        count_map[b] = flat_count.view(1, H, W)
        
    # 取平均
    mask = count_map > 0
    error_map[mask] /= count_map[mask]
    
    # 3. 扩散 (Dilation / Blurring)
    # 因为点很稀疏，图像会有很多空洞，需要用 MaxPool "扩散" 误差，填补空隙
    # Kernel size 决定了扩散半径，取决于点云的稀疏程度
    error_map = F.max_pool2d(error_map, kernel_size=5, stride=1, padding=2)
    # 再稍微平滑一下
    # error_map = F.avg_pool2d(error_map, kernel_size=3, stride=1, padding=1)
    
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

        self.blur = KNNGaussianBlur(4)
        
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

    def compute_hybrid_loss(self, pred, target, dim=-1):
        """
        混合损失函数：Cosine (主) + L2 (辅)
        """
        # 1. Cosine Loss (主): 1 - cos_sim
        # pred, target: [B, N, C] or [B, C, H, W]
        # ★ 终极防线 3：强制加上极小值，防止纯黑图像/零特征引发 NaN
        pred_safe = pred + 1e-8
        target_safe = target + 1e-8
        cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim).mean()
        
        # 2. L2 Loss (辅): MSE
        l2_loss = F.mse_loss(pred, target)
        
        return cosine_loss + self.l2_weight * l2_loss

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
        rgb_recon, xyz_recon, rgb_target, F_gt, loss_geo, U, mask, center = self.net(rgb, xyz, ps)

        # -------------------------
        # 计算 Loss
        # -------------------------
        
        # 1. 2D Loss: MSE(Frgb, Frgb')
        # 文档: "2d分支的损失函数是 Frgb 和 Frgb, 之间的误差"
        # 2D (Channel维度是 dim=1)
        loss_2d = self.compute_hybrid_loss(rgb_recon, rgb_target.detach(),dim = 1) 
        # 注意: 通常 Target 不需要梯度，detach 掉 rgb_target 以防止梯度传回 Encoder
        
        # 2. 3D Loss: Global MSE Loss
        # 文档: "Lrec = |Fpred - Fgt|^2"
        # 3D (Channel维度是 dim=2)
        loss_3d = self.compute_hybrid_loss(xyz_recon, F_gt.detach(),dim = 2)
        
        # 3. ★ OT 计算权重 (传入 detach 防止作弊)
        # 只有在训练时，我们希望"避重就轻"，让模型先学容易的，稳步收敛
        alpha, beta = self.ot_module(loss_2d.detach(), loss_3d.detach())
        
        # 取 Batch 平均权重进行反向传播
        w_alpha = alpha.mean()
        w_beta = beta.mean()
    
        weighted_loss = w_alpha * loss_2d + w_beta * loss_3d
        total_loss = weighted_loss + self.lambda_geo * loss_geo
        
        return {
            "loss": total_loss,
            "l2d": loss_2d.item(),
            "l3d": loss_3d.item(),
            "geo": loss_geo.item(),
            "alpha": w_alpha.item(),
            "beta": w_beta.item()
        }

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
        # depth_map = depth_map.to(self.device)
        ps = ps.to(self.device)
        B = rgb.size(0)

        # -------------------------
        # 3. 前向重建（注意要把 ps 也传进去）
        # -------------------------
        rgb_recon, xyz_recon, rgb_target, F_gt, _, _, _, center = self.net(rgb, xyz, ps)
        # rgb_recon: [B, 3, H, W]
        # xyz_recon: [B, 1, H, W]
        # === 调试打印 ===
        print(f"Center Range - X: [{center[..., 0].min().item():.2f}, {center[..., 0].max().item():.2f}]")
        print(f"Center Range - Y: [{center[..., 1].min().item():.2f}, {center[..., 1].max().item():.2f}]")

        # -------------------------
        # 4. 重建误差 map
        # -------------------------
        # ---------------------------------------------------
        # 步骤 A: 计算误差数值 (L2 Norm + Euclidean)
        # ---------------------------------------------------
        
        # 1. L2 归一化 (解决量级差异的关键)
        rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)   # [B, C, H, W]
        rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
        
        xyz_recon_norm = F.normalize(xyz_recon, p=2, dim=2)   # [B, N, C]
        F_gt_norm = F.normalize(F_gt, p=2, dim=2)

        # 2. 计算欧氏距离平方 (逐像素/逐点)
        # 2D Error: [B, H, W]
        err_2d = torch.sum((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
        
        # 3D Error Points: [B, N] (这是 N 个离散点的误差值)
        err_3d_points = torch.sum((xyz_recon_norm - F_gt_norm) ** 2, dim=2)
        
        # ---------------------------------------------------
        # 步骤 B: 空间映射 (Splatting) - 这一步绝对不能省！
        # ---------------------------------------------------
        
        # 利用 center 坐标，将离散的 err_3d_points 投影回 2D 网格
        # 如果没有这一步，3D 误差图就是乱序的噪声
        err_3d_map = splat_3d_error_to_2d(err_3d_points, center, self.img_size_val)

        # ---------------------------------------------------
        # 步骤 C: 后处理与融合
        # ---------------------------------------------------
        
        # 2D 上采样 (如果 backbone 输出不是 224)
        if err_2d.shape[-1] != self.img_size_val:
            err_2d = F.interpolate(err_2d.unsqueeze(1), size=(self.img_size_val, self.img_size_val), 
                                  mode='bilinear', align_corners=False).squeeze(1)
        
        # 高斯模糊 (平滑噪声)
        err_2d_smooth = self.blur(err_2d.unsqueeze(1).cpu()).squeeze(1).to(self.device)
        err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).squeeze(1).to(self.device)

        # 5. ★ 推理融合：直接相加 (等价于 alpha=0.5, beta=0.5)
        # 这样任何一个分支检测到的缺陷 (高误差) 都会被保留
        fused = err_2d_smooth + err_3d_smooth

        # -------------------------
        # 7. 仿 RealIAD-D3：把 score 累积到 buffer 里
        # -------------------------
        gt = gt.detach().cpu()
        label = label.detach().cpu()
        fused = fused.cpu()
        for b in range(B):
            gmap = gt[b, 0]           # [Hg, Wg]
            fmap = fused[b, 0]          # [1,H,W] / [H,W] / [1,L] / [L]

            # ========= 现在 fmap / gmap 都是 [Hg, Wg] =========
            img_score = float(fmap.max())
            self.image_preds.append(img_score)
            self.image_labels.append(int(label[b]))

            # 存展平的像素 (慎用，内存大)
            self.pixel_preds.extend(fmap.flatten().numpy().tolist())
            self.pixel_labels.extend(gmap.flatten().numpy().tolist())

            self.gts.append(gmap.numpy())
            self.pred_maps.append(fmap.clone())
            
            # --- 保存单独的 Map ---
            self.maps_2d.append(err_2d_smooth[b].clone())
            self.maps_3d.append(err_3d_smooth[b].clone())

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

        # 像素级 ROC
        self.pixel_rocauc = float(roc_auc_score(self.pixel_labels, self.pixel_preds))

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




