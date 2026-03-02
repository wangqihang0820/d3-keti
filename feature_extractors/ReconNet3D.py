import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from models.models import Block
import math


# 序列长度为1024，Point Transformer V1/V2：需要做局部 k-NN 建图，在 1024 个点上做局部建图不如直接做全局 Attention 来得直接和有效。故直接用的model里的block，没用标准的Point Transformer V1 (Zhao et al.)"、"V2" 或 "V3"
class TransitionDown(nn.Module):
    """
    对应图中 Encoder 里的 Transition Down
    作用: 通道翻倍，分辨率减半 (序列长度 1/4)
    实现: Reshape成2D -> Conv stride 2 -> Reshape回序列
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, G, C]
        B, G, C = x.shape
        H = int(math.sqrt(G))
        W = H
        
        # [B, G, C] -> [B, C, H, W]
        x_img = x.transpose(1, 2).view(B, C, H, W)
        
        # Downsample
        x_down = self.conv(x_img) # [B, OutC, H/2, W/2]
        
        # Back to Sequence: [B, OutC, H/2, W/2] -> [B, G/4, OutC]
        x_out = x_down.flatten(2).transpose(1, 2)
        x_out = self.norm(x_out)
        
        return x_out

class TransitionUp(nn.Module):
    """
    对应图中 Decoder 里的 Transition Up
    作用: 通道减半，分辨率翻倍 (序列长度 x4)
    实现: Reshape成2D -> ConvTranspose stride 2 -> Reshape回序列
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, G, C]
        B, G, C = x.shape
        H = int(math.sqrt(G))
        W = H
        
        # [B, G, C] -> [B, C, H, W]
        x_img = x.transpose(1, 2).view(B, C, H, W)
        
        # Upsample
        x_up = self.up(x_img) # [B, OutC, H*2, W*2]
        
        # Back to Sequence
        x_out = x_up.flatten(2).transpose(1, 2)
        x_out = self.norm(x_out)
        
        return x_out

class SkipFusion(nn.Module):
    """
    对应图中的 Skip Connection 结构:
    Dropout -> Concat -> MLP
    """
    def __init__(self, dim, drop_rate=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop_rate)
        # Concat后通道数翻倍，通过MLP降维回 dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x_enc, x_dec):
        """
        x_enc: 来自 Encoder 的特征 (经过 Dropout)
        x_dec: 来自 Decoder 上一层的特征 (经过 Upsample)
        """
        # 图中 Dropout 加在 Encoder 特征上
        x_enc = self.drop(x_enc)
        
        # Concat (C)
        x_cat = torch.cat([x_dec, x_enc], dim=-1) # [B, G, 2C]
        
        # MLP Fusion
        x_out = self.mlp(x_cat) # [B, G, C]
        
        return x_out

class ReconNet3D(nn.Module):
    """
    严格对应 Image 2 (3D Reconstruction Network) 的架构
    Input: [B, G, C] (e.g., [B, 1024, 768])
    """
    def __init__(self, in_dim=768, base_dim=384):
        super().__init__()
        
        # ===========================
        # Encoder
        # ===========================
        # 图中 Encoder 有三层 Block，两次 Down
        
        # --- Stage 1 ---
        # 1. Point Transformer Block
        self.enc_block1 = Block(in_dim, num_heads=6)
        # 2. Transition Down (768 -> 768*2) (通道翻倍，点数减少)
        # 注意: 通常 U-Net 下采样通道翻倍。这里假设 base_dim 是中间维度，或者我们直接按 768->768 走也可以。
        # 为了构建强 Bottleneck，建议: 768 -> 384 (降维) 或 768 -> 768 (保持)。
        # 你的图中没有标维度，按照 PointMAE 输出 768，我们可以设计为:
        # L1: 768 -> Down -> L2: 1536 -> Down -> L3: 3072 (太大了)
        # 或者: 
        # L1: 768 (1024点) -> Down -> L2: 768 (256点) -> Down -> L3: 768 (64点)
        # 鉴于显存，建议保持通道数 768 不变，只变序列长度。
        
        self.dim = in_dim # 768
        
        self.trans_down1 = TransitionDown(self.dim, self.dim)
        
        # --- Stage 2 ---
        self.enc_block2 = Block(self.dim, num_heads=6)
        self.trans_down2 = TransitionDown(self.dim, self.dim)
        
        # --- Bottleneck (Encoder End / Decoder Start) ---
        self.bottleneck_block = Block(self.dim, num_heads=6)
        
        # 图中间的一条线: ReLU (虽然 Transformer 内部有激活，但图中显式画了)
        self.mid_relu = nn.ReLU()
        
        # ===========================
        # Decoder
        # ===========================
        
        # --- Stage 2 Decode ---
        self.trans_up2 = TransitionUp(self.dim, self.dim)
        self.fusion2 = SkipFusion(self.dim) # 处理 Enc2 和 Up2 的融合
        self.dec_block2 = Block(self.dim, num_heads=6)
        
        # --- Stage 1 Decode ---
        self.trans_up1 = TransitionUp(self.dim, self.dim)
        self.fusion1 = SkipFusion(self.dim) # 处理 Enc1 和 Up1 的融合
        self.dec_block1 = Block(self.dim, num_heads=6)
        
        # Output Head
        # 图中 Output 只是一个框，通常加一个 Linear 调整
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        # x: [B, 1024, 768]
        
        # ===================
        # Encoder
        # ===================
        
        # Stage 1
        x_e1 = self.enc_block1(x)       # [B, 1024, 768] -> Skip Source 1
        x_d1 = self.trans_down1(x_e1)   # [B, 256, 768]
        
        # Stage 2
        x_e2 = self.enc_block2(x_d1)    # [B, 256, 768] -> Skip Source 2
        x_d2 = self.trans_down2(x_e2)   # [B, 64, 768]
        
        # Bottleneck
        x_bot = self.bottleneck_block(x_d2) # [B, 64, 768]
        
        # Mid ReLU (Connection)
        x_mid = self.mid_relu(x_bot)
        
        # ===================
        # Decoder
        # ===================
        
        # Stage 2 (对应 Encoder Stage 2)
        # 1. Transition Up
        x_u2 = self.trans_up2(x_mid)    # [B, 256, 768]
        # 2. Skip Fusion (Dropout -> Concat -> MLP)
        x_f2 = self.fusion2(x_e2, x_u2) # Enc2 + Up2
        # 3. Block
        x_dec2 = self.dec_block2(x_f2)  # [B, 256, 768]
        
        # Stage 1 (对应 Encoder Stage 1)
        # 1. Transition Up
        x_u1 = self.trans_up1(x_dec2)   # [B, 1024, 768]
        # 2. Skip Fusion
        x_f1 = self.fusion1(x_e1, x_u1) # Enc1 + Up1
        # 3. Block
        x_out = self.dec_block1(x_f1)   # [B, 1024, 768]
        
        # Output
        out = self.out_proj(x_out)
        
        return out