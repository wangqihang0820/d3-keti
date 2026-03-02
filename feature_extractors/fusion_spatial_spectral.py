import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusionLayer(nn.Module):
    """
    Step 5.2: 自适应注入门控 (Adaptive Injection Gate) - GLU 改进版
    参考逻辑: F_sum -> Linear(2C) -> Chunk -> GELU(x1) * x2 -> F_inject
    """
    def __init__(self, channels=768):
        super().__init__()
        
        # 1. 映射层 (对应参考代码中的 dwconv 作用，但适配点云)
        # 将维度扩展为 2倍，以便后续切分为 x1 和 x2
        # 这里使用 Linear 等价于 kernel=1 的 Conv1d，专注于通道混合和特征选择
        self.proj = nn.Linear(channels, channels * 2)
        
        # 2. 最终融合层 (可选，用于稳定输出分布)
        # 将 GLU 的输出再次整理，确保与 F_in 处于同一特征空间
        self.out_proj = nn.Linear(channels, channels)
        
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, f_sum):
        """
        Args:
            f_sum: [B, N, C] 双流汇聚后的特征
        Returns:
            f_inject: [B, N, C] 准备注入回 F_in 的增量特征
        """
        # 1. 扩展维度 [B, N, C] -> [B, N, 2C]
        x = self.proj(f_sum)
        
        # 2. 分离 (Chunk)
        # x1: 门控/激活分支, x2: 信息分支
        x1, x2 = x.chunk(2, dim=-1) # [B, N, C], [B, N, C]
        
        # 3. GLU 机制 (Gated Linear Unit)
        # 参考代码逻辑: F.gelu(x1) * x2
        # GELU 作为激活函数，x2 作为原本的信息流，两者相乘实现"软门控"
        # 这种结构允许网络选择性地通过信息，同时引入非线性
        f_gate = F.gelu(x1) * x2
        
        # 4. 输出投影
        f_inject = self.out_proj(f_gate)
        
        return f_inject


class DualStreamFusion(nn.Module):
    """
    Phase 5: 融合与重建 (Dual-Stream Fusion & Reconstruction)
    Step 5.1: 双流汇聚 (Element-wise Sum)
    Step 5.2: GLU 自适应门控注入
    """
    def __init__(self, channels=768):
        super().__init__()
        self.gate_layer = GatedFusionLayer(channels=channels)

    def forward(self, f_freq_out, f_spatial_out, f_in):
        """
        Args:
            f_freq_out:    [B, N, C] 频域分支输出 (GGRM output)
            f_spatial_out: [B, N, C] 空间分支输出 (Spatial Branch output)
            f_in:          [B, N, C] 原始输入 (Step 2.2 的输出, 含 Mask Token)
                           注意: 这是"残差回注"的基座
        Returns:
            f_out:         [B, N, C] 最终重建特征
        """
        
        # ===============================================
        # Step 5.1: 双流特征汇聚 (Dual-Stream Aggregation)
        # ===============================================
        # 简单的加和，让每个点同时拥有 频域(全局几何) 和 空间(局部细节) 信息
        f_sum = f_freq_out + f_spatial_out
        
        # ===============================================
        # Step 5.2: 自适应注入门控 (Adaptive Injection Gate)
        # ===============================================
        # 使用 GLU 机制计算增量
        f_inject = self.gate_layer(f_sum)
        
        # 残差回注 (Residual Injection)
        # F_out = F_in + G(F_sum)
        # 对于 Mask 区域: F_in 是 mask token, 网络会利用 f_inject 填补信息
        # 对于 Unmask 区域: F_in 是真值, 网络通常会让 f_inject 较小，或者是作为一种 Refine
        f_out = f_in + f_inject
        
        return f_out