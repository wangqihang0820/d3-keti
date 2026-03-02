import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

class EdgeConvLayer(nn.Module):
    """
    标准的 EdgeConv 模块 (Dynamic Graph CNN 核心组件)
    流程: Grouping -> Edge Construction -> MLP -> Max Pooling
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 3. 特征变换 (Internal MLP)
        # 输入维度: 2 * C (中心点特征 + 差异特征)
        # 结构: Linear(2C -> C') -> BN -> ReLU
        # 使用 Conv2d(kernel=1) 实现高效的 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, idx):
        """
        Args:
            x:   [B, N, C] 输入特征
            idx: [B, N, k] k-NN 索引 (复用 Step 1.2 的结果)
        Returns:
            out: [B, N, C'] 聚合后的特征
        """
        B, N, C = x.shape
        k = idx.shape[2]

        # ---------------------------------------------------
        # 1. 分组 (Grouping)
        # ---------------------------------------------------
        # 转换维度以适配 pointnet2_utils: [B, N, C] -> [B, C, N]
        x_trans = x.transpose(1, 2).contiguous()
        
        # 收集邻居特征 F_neighbors
        # 输入: x=[B, C, N], idx=[B, N, k]
        # 输出: [B, C, N, k]
        neighbor_x = pointnet2_utils.grouping_operation(x_trans, idx.int())
        
        # 准备中心点特征 (扩展维度以便拼接)
        # [B, C, N] -> [B, C, N, 1] -> [B, C, N, k]
        center_x = x_trans.unsqueeze(-1).expand(-1, -1, -1, k)

        # ---------------------------------------------------
        # 2. 边特征构造 (Edge Feature Construction)
        # ---------------------------------------------------
        # e_ij = Concat(x_i, x_j - x_i)
        # [B, C, N, k] cat [B, C, N, k] -> [B, 2C, N, k]
        edge_feat = torch.cat([center_x, neighbor_x - center_x], dim=1)

        # ---------------------------------------------------
        # 3. 特征变换 (Internal MLP)
        # ---------------------------------------------------
        # [B, 2C, N, k] -> [B, C', N, k]
        edge_feat_transformed = self.mlp(edge_feat)

        # ---------------------------------------------------
        # 4. 聚合 (Aggregation)
        # ---------------------------------------------------
        # Max Pooling over neighbors (dim=-1)
        # [B, C', N, k] -> [B, C', N]
        out, _ = torch.max(edge_feat_transformed, dim=-1)

        # 转回常用维度: [B, N, C']
        out = out.transpose(1, 2).contiguous()
        
        return out


class SpatialComplementBranch(nn.Module):
    """
    Phase 4: 空间互补分支 (Spatial Complement Branch)
    结构: EdgeConv -> GELU -> EdgeConv
    作用: 捕获局部微小几何不连续性，填补频域分支的细节缺失
    """
    def __init__(self, channels=768):
        super().__init__()
        
        # 1. 第一层 EdgeConv (EdgeConv Layer 1)
        # 输入 768 -> 输出 768
        self.edge_conv1 = EdgeConvLayer(in_channels=channels, out_channels=channels)
        
        # 2. 中间激活 (Intermediate Activation)
        # 使用 GELU 增加平滑非线性
        self.act = nn.GELU()
        
        # 3. 第二层 EdgeConv (EdgeConv Layer 2)
        # 输入 768 -> 输出 768 (扩大感受野)
        self.edge_conv2 = EdgeConvLayer(in_channels=channels, out_channels=channels)

    def forward(self, f_in, knn_idx):
        """
        Args:
            f_in:    [B, N, C] Masked Features (来自 Step 2.2)
            knn_idx: [B, N, k] k-NN 索引 (来自 Step 1.2)
        Returns:
            f_spatial_out: [B, N, C] 空间互补特征
        """
        # EdgeConv 1
        x = self.edge_conv1(f_in, knn_idx)
        
        # GELU Activation
        x = self.act(x)
        
        # EdgeConv 2
        f_spatial_out = self.edge_conv2(x, knn_idx)
        
        return f_spatial_out