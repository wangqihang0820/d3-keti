import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class LatentRandomMasking(nn.Module):
    """
    Step 2.2: 潜在空间随机特征掩码 (Latent Random Feature Masking)
    机制: Mask-and-Replace (保留序列长度 M，用可学习 Token 替换特征)
    """
    def __init__(self, input_dim=768, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_dim = input_dim

        # 【核心差异】
        # PointMAE 是丢弃 Token，我们这里是替换 Token。
        # 定义一个可学习的特征向量 [MASK]，用于填充被遮挡的位置。
        # 它的作用类似于 BERT 里的 [MASK] 词向量。
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        # 初始化
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x):
        """
        输入: x [B, M, C] (Latent Ground Truth F_gt)
        输出: 
            x_masked [B, M, C] (Corrupted Input F_in)
            mask_bool [B, M]   (0: visible, 1: masked)
        """
        B, M, C = x.shape
        device = x.device

        # -------------------------------------------------
        # 1. 生成随机掩码 (Random Masking)
        # -------------------------------------------------
        # 这里的 "噪声+排序" 是实现 "无放回随机采样" 的最高效方式
        noise = torch.rand(B, M, device=device)
        
        # argsort 返回的是索引，相当于给 Token 随机洗牌
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # 计算要掩盖的数量 (例如 1024 * 0.6 = 614)
        len_mask = int(M * self.mask_ratio)
        
        # 前 len_mask 个索引被选中进行掩盖
        mask_indices = ids_shuffle[:, :len_mask]
        
        # 创建 mask 矩阵 (1表示被掩盖)
        mask_bool = torch.zeros(B, M, device=device)
        mask_bool.scatter_(1, mask_indices, 1.0) # [B, M]

        # -------------------------------------------------
        # 2. 特征替换 (Feature Replacement)
        # -------------------------------------------------
        mask_expanded = mask_bool.unsqueeze(-1) # [B, M, 1]

        # 公式: F_in = F_gt * (1 - Mask) + Mask_Token * Mask
        # 可见部分保持原样，掩盖部分替换为 learnable mask token
        x_masked = x * (1 - mask_expanded) + self.mask_token * mask_expanded

        return x_masked, mask_bool