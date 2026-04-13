import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusionLayer(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.proj = nn.Linear(channels, channels * 2)
        self.out_proj = nn.Linear(channels, channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, f_sum):
        x = self.proj(f_sum)
        x1, x2 = x.chunk(2, dim=-1)
        f_gate = F.gelu(x1) * x2
        f_inject = self.out_proj(f_gate)
        return f_inject

class DualStreamFusion(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.gate_layer = GatedFusionLayer(channels=channels)
        # ★ 新增：可学习的残差门控，初始化为 0 (Sigmoid后为0.5)
        self.residual_gate = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, f_freq_out, f_spatial_out, f_in):
        f_sum = f_freq_out + f_spatial_out
        f_inject = self.gate_layer(f_sum)
        
        # ★ 新增：软门控衰减，强制网络利用双流修复信息而非原样输出 F_in
        alpha = torch.sigmoid(self.residual_gate)
        f_out = f_in * alpha + f_inject
        
        return f_out
