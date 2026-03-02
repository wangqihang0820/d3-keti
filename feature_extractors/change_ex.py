import torch
import torch.nn as nn

# 用于在两个特征图（x0 和 x1）之间交换通道，通常用于时间序列数据或双时态特征融合。
class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        # p 参数决定了交换的通道数量比例。具体来说，将会交换 1/p 的特征通道。
        self.p = p  # 1/p of the features will be exchanged.

# x0 和 x1 是输入的特征图，形状为 (N, C, H, W)，其中：N：批量大小 C：通道数 H：特征图的高度 W：特征图的宽度
    def forward(self, x0, x1):
        # x0, x1: the bi-temporal feature maps.
        N, C, H, W = x0.shape
        # 创建一个 exchange_mask，它是一个布尔掩码，表示哪些通道会被交换
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        
        out_x0 = torch.zeros_like(x0)
        out_x1 = torch.zeros_like(x1)

        out_x0[~exchange_mask] = x0[~exchange_mask]
        out_x1[~exchange_mask] = x1[~exchange_mask]
        out_x0[exchange_mask] = x1[exchange_mask]
        out_x1[exchange_mask] = x0[exchange_mask]

        return out_x0, out_x1
    
# SpatialExchange 类用于在两个特征图之间进行空间维度（宽度方向）的交换    
class SpatialExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p  # 1/p of the features will be exchanged.

    def forward(self, x0, x1):
        # x0, x1: the bi-temporal feature maps.
        N, C, H, W = x0.shape
        # Create a mask based on width dimension
        exchange_mask = torch.arange(W, device=x0.device) % self.p == 0
        # Expand mask to match feature dimensions
        exchange_mask = exchange_mask.view(1, 1, 1, W).expand(N, C, H, -1)

        out_x0 = x0.clone()
        out_x1 = x1.clone()

        # Perform column-wise exchange
        out_x0[..., exchange_mask] = x1[..., exchange_mask]
        out_x1[..., exchange_mask] = x0[..., exchange_mask]

        return out_x0, out_x1

# CombinedExchange 类结合了 ChannelExchange 和 SpatialExchange，在输入特征图上同时执行通道交换和空间交换。
class CombinedExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.channel_exchange = ChannelExchange(p=p)
        self.spatial_exchange = SpatialExchange(p=p)

    def forward(self, x0, x1):
        # First perform channel exchange
        x0, x1 = self.channel_exchange(x0, x1)
        # Then perform spatial exchange
        x0, x1 = self.spatial_exchange(x0, x1)
        return x0, x1
