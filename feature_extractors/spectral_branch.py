import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# 高低频分离
class SpectralTransform(nn.Module):
    """
    Step 3.1: 全频域变换 (Full Frequency Domain Transformation)
    功能: 负责空间域(Spatial)与谱域(Spectral)之间的相互变换
    """
    def __init__(self):
        super().__init__()

    def gft(self, f_spatial, U):
        """
        图傅里叶变换 (Graph Fourier Transform)
        Formula: F_spec = U^T * F_spatial
        
        Args:
            f_spatial: [B, N, C] 空间域特征 (F_in)
            U: [B, N, N] 正交基底 (Eigenvectors)
        Returns:
            f_spec: [B, N, C] 谱域特征
        """
        # U.transpose: [B, N, N]
        # matmul: [B, N, N] x [B, N, C] -> [B, N, C]
        f_spec = torch.matmul(U.transpose(1, 2), f_spatial)
        return f_spec

    def igft(self, f_spec, U):
        """
        图傅里叶逆变换 (Inverse Graph Fourier Transform)
        Formula: F_spatial = U * F_spec
        
        Args:
            f_spec: [B, N, C] 谱域特征
            U: [B, N, N] 正交基底
        Returns:
            f_spatial: [B, N, C] 恢复的空间域特征
        """
        # matmul: [B, N, N] x [B, N, C] -> [B, N, C]
        f_spatial = torch.matmul(U, f_spec)
        return f_spatial


# v34以及之后的   基于能量比例的动态硬切分门控
class DynamicContentGating(nn.Module):
    """
    Step 3.2: 基于能量比例的动态硬切分门控 (Energy-Proportional Hard Gating with STE)
    核心思想: 动态预测一个能量比例(如90%)，将累积能量达到该比例之前的所有频率划为低频，其余为高频。
    """
    def __init__(self, num_points=1024, reduction=16):
        super().__init__()
        
        hidden_dim = max(num_points // reduction, 16)
        
        # 此时的 MLP 不再预测 1024 个频率的独立门控，
        # 而是预测当前工件的"目标能量保留比例 (Target Ratio)"
        self.mlp = nn.Sequential(
            nn.Linear(num_points, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1) # 输出一个标量 logits
        )
        
        # 设置一个初始偏置，例如初始化为 2.0
        # sigmoid(2.0) ≈ 0.88，意味着训练初期默认保留前 88% 的能量作为低频骨架
        self.bias_ratio = nn.Parameter(torch.tensor([2.0])) 

    def forward(self, f_spec):
        """
        Args:
            f_spec: [B, M, C] 全频段特征 
            (注: torch.linalg.eigh 输出的特征向量已经天然按照频率从低到高排列)
        """
        B, M, C = f_spec.shape
        
        # ----------------------------------------------------
        # 1. 频域能量计算与累积 (Energy Accumulation)
        # ----------------------------------------------------
        # 计算每个频率分量的绝对能量 [B, M]
        energy = f_spec.abs().mean(dim=-1) 
        
        # 计算总能量 [B, 1]
        total_energy = energy.sum(dim=-1, keepdim=True) + 1e-6 
        
        # 计算从低频到高频的累积能量 [B, M]
        cum_energy = torch.cumsum(energy, dim=-1)
        
        # 计算累积能量占比 (递增序列，从 >0 到 1.0) [B, M]
        cum_ratio = cum_energy / total_energy 
        
        # ----------------------------------------------------
        # 2. 预测目标能量比例 (Target Ratio Prediction)
        # ----------------------------------------------------
        # MLP 根据全局能量分布预测 logits，再加上物理先验偏置
        logits_ratio = self.mlp(energy) + self.bias_ratio # [B, 1]
        
        # 转换为 0~1 之间的目标比例 (如 0.85) [B, 1]
        target_ratio = torch.sigmoid(logits_ratio) 
        
        # ----------------------------------------------------
        # 3. 基于比例的硬切分与 STE 魔法
        # ----------------------------------------------------
        # Hard Gate: 累积占比 <= 目标比例的，设为 1 (低频)；超过的设为 0 (高频)
        hard_gate = (cum_ratio <= target_ratio).float() # [B, M]
        
        # Soft Gate (用于提供梯度): 
        # 使用平滑的 Sigmoid 阶跃函数。当 cum_ratio < target_ratio 时趋于1，反之趋于0
        # 放大系数 alpha=10 控制梯度平滑度
        soft_gate = torch.sigmoid(10.0 * (target_ratio - cum_ratio)) # [B, M]
        
        # STE 直通估计器：前向是绝对硬切分，反向通过 soft_gate 传递梯度给 target_ratio
        gate = hard_gate.detach() - soft_gate.detach() + soft_gate # [B, M]
        
        # 扩展维度 [B, M, 1]
        gate_expanded = gate.unsqueeze(-1)
        
        # ----------------------------------------------------
        # 4. 正交分离
        # ----------------------------------------------------
        f_low = f_spec * gate_expanded
        f_high = f_spec * (1.0 - gate_expanded)
        
        # 测试阶段可以选择性打印 target_ratio 查看网络学到的切分比例
        # if not self.training:
        #     print(f"当前样本的动态截断能量比例: {target_ratio.mean().item():.3f}")
        
        return f_low, f_high, gate_expanded
      
      
      
      
# 低频分支
class SpectralFingerprint(nn.Module):
    """
    1. 频率指纹提取
    公式: Z = GAP(F_low) + GMP(F_low)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, M, C]
        # 在频率维度 M (dim=1) 上进行池化
        gap = torch.mean(x, dim=1) # [B, C]
        gmp, _ = torch.max(x, dim=1) # [B, C]
        z = gap + gmp
        return z

class UncertaintyRouter(nn.Module):
    """
    2. 双分支不确定性路由 (Gate)
    替代了 lfmoe 中的 GateNetwork，改为文档要求的双分支 MLP 结构
    """
    def __init__(self, input_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

        # 信号预测分支 (Signal Branch): MLP -> Linear -> S_pred
        self.mlp_signal = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, num_experts)
        )

        # 不确定性噪声分支 (Noise Branch): MLP -> Linear -> Softplus -> sigma
        self.mlp_noise = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 4, num_experts),
            nn.Softplus()
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        # z: [B, C]
        
        # 1. 计算信号得分
        s_pred = self.mlp_signal(z) # [B, N_experts]

        # 2. 计算不确定性
        sigma = self.mlp_noise(z)   # [B, N_experts]

        # 3. 重参数化采样 (只在训练时加噪声)
        if self.training:
            epsilon = torch.randn_like(sigma)
            s_final = s_pred + sigma * epsilon
        else:
            s_final = s_pred

        # 4. Top-K 筛选
        topk_values, indices = torch.topk(s_final, k=self.top_k, dim=1)

        # 5. 权重归一化 (Softmax 只作用于 Top-K)
        mask_logits = torch.full_like(s_final, float('-inf'))
        mask_logits.scatter_(1, indices, topk_values)
        routing_weights = self.softmax(mask_logits)

        return routing_weights, indices

class SpectralExpert(nn.Module):
    """
    4. 同质专家 (Homogeneous Expert)
    结构: 1D Spectral Convolution (单层滤波，无残差)
    配置: Kernel=3, Padding=1, Groups=C
    """
    def __init__(self, channels):
        super().__init__()
        # Depthwise Conv1d: 每个通道独立进行频域滤波
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels, # 关键：Groups=C
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化
        init.kaiming_normal_(self.conv1d.weight, mode='fan_in')
        init.constant_(self.conv1d.bias, 0.0)

    def forward(self, x):
        # x: [B, C, M] (注意输入需要转置)
        out = self.conv1d(x)
        return self.relu(out)

class FD_MoE(nn.Module):
    """
    Step 3.3: 谱域自适应混合专家完整模块
    """
    def __init__(self, channels=768, num_experts=4, top_k=2):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        
        # 组件实例化
        self.fingerprint = SpectralFingerprint()
        self.router = UncertaintyRouter(channels, num_experts, top_k)
        self.experts = nn.ModuleList([
            SpectralExpert(channels) for _ in range(num_experts)
        ])

    def inverse_transform(self, f_low_prime, U):
        """
        Step 3.3.5: 低频逆变换
        F_out = U * F_prime
        """
        # U: [B, M, M], f_low_prime: [B, M, C]
        return torch.matmul(U, f_low_prime)

    def forward(self, f_low, U):
        """
        f_low: [B, M, C] (来自 CBDG 的低频特征)
        U: [B, M, M]
        """
        B, M, C = f_low.shape

        # 1. 提取指纹 [B, C]
        z = self.fingerprint(f_low)

        # 2. 路由计算 [B, N]
        weights, _ = self.router(z)

        # 3. 混合专家计算
        # Conv1d 需要 [B, C, M]
        x_in = f_low.transpose(1, 2)
        
        final_out = torch.zeros_like(x_in)

        # 遍历专家 (加权求和)
        for i in range(self.num_experts):
            expert_w = weights[:, i].view(-1, 1, 1) # [B, 1, 1]
            
            # 只有当该专家有权重时才计算 (虽然 batch 下可能还是都会算)
            if expert_w.sum() > 0:
                expert_out = self.experts[i](x_in)
                final_out += expert_w * expert_out

        # 转回 [B, M, C]
        f_low_prime = final_out.transpose(1, 2)

        # 4. 逆变换回空间域
        f_low_spatial = self.inverse_transform(f_low_prime, U)

        return f_low_spatial


# 高频分支
class GGRM(nn.Module):
    """
    Step 3.4: 几何引导精修 (Geometry-Guided Refinement Module)
    & Step 3.5: 频域重组 (Frequency Reconstruction)
    
    包含:
    1. 高频谱域预处理 (Global Filtering)
    2. 高频逆变换 (IGFT)
    3. 双分支注意力门控 (Dual-Branch Attention)
    4. 特征重组 (Reconstruction)
    """
    def __init__(self, channels=768, num_points=1024, reduction=4):
        """
        Args:
            channels: 特征维度 (d_model), 默认 768
            num_points: 频率点数 (M), 默认 1024
            reduction: MLP 中间层的缩放比例, 用于降低参数量
        """
        super().__init__()
        
        # -------------------------------------------------------
        # 1. 高频预处理参数 (Spectral Weighting)
        # -------------------------------------------------------
        # W: [M, 1] 可学习的频域加权参数
        # 作用: 对 1024 个频率分量进行全局加权 (抑制极高频噪声)
        # 形状 [1, M, 1] 以便广播: [B, M, C] * [1, M, 1]
        self.high_freq_weight = nn.Parameter(torch.ones(1, num_points, 1))
        # 初始化为 0.5 或 1.0，允许网络学习哪些频率该保留
        init.normal_(self.high_freq_weight, mean=0.5, std=0.02)

        # -------------------------------------------------------
        # 2. GGRM 双分支注意力 (Dual-Branch Attention)
        # -------------------------------------------------------
        # 参考您的 MSCW 代码结构，改为: Linear -> ReLU -> Linear
        
        # 分支一: 全局通道感知 (Global Channel Context)
        # 输入: GAP后的特征 [B, 1, C]
        self.global_attn = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )

        # 分支二: 局部点级感知 (Local Point-wise Context)
        # 输入: 原始特征 [B, M, C] (点对点处理)
        self.local_attn = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )

        self.sigmoid = nn.Sigmoid()

        # -------------------------------------------------------
        # 3. 频域重组 (Step 3.5)
        # -------------------------------------------------------
        # 输入: Concat(Low, High_Refined) -> 2 * channels
        self.fusion_linear = nn.Linear(channels * 2, channels)

    def forward(self, f_low_spatial, f_high_spectral, U):
        """
        Args:
            f_low_spatial:   [B, M, C] 也就是 Guide, 来自 FD-MoE 输出 (空间域)
            f_high_spectral: [B, M, C] 也就是 Target, 来自 CBDG 输出 (谱域)
            U:               [B, M, M] 正交基底, 用于 IGFT
        Returns:
            f_out:           [B, M, C] 最终重组后的特征
        """
        B, M, C = f_low_spatial.shape

        # ===============================================
        # Part A: 高频预处理 & 逆变换
        # ===============================================
        
        # 1. 全局滤波 (Spectral Filtering)
        # F_high_prime = F_high * W
        f_high_prime = f_high_spectral * self.high_freq_weight

        # 2. 高频逆变换 (IGFT)
        # F_high_spatial = U * F_high_prime
        # [B, M, M] @ [B, M, C] -> [B, M, C]
        f_high_spatial = torch.matmul(U, f_high_prime)

        # ===============================================
        # Part B: GGRM 核心逻辑 (MSCM 改进版)
        # ===============================================

        # 3. 特征预融合 (Feature Pre-fusion)
        # F_mix = F_low + F_high (空间域相加)
        f_mix = f_low_spatial + f_high_spatial

        # 4. 双分支注意力生成
        # Branch 1: Global (GAP -> MLP)
        # g = GAP(F_mix) -> [B, 1, C]
        g = torch.mean(f_mix, dim=1, keepdim=True)
        a_global = self.global_attn(g) # [B, 1, C]

        # Branch 2: Local (Point-wise MLP)
        # [B, M, C] -> MLP -> [B, M, C]
        a_local = self.local_attn(f_mix) # [B, M, C]

        # 5. 门控融合 & 生成
        # H = A_global + A_local (广播机制)
        h = a_global + a_local
        m_refine = self.sigmoid(h) # [B, M, C]

        # 6. 精修执行
        # F_high_refined = F_high_spatial * M_refine
        f_high_refined = f_high_spatial * m_refine

        # ===============================================
        # Part C: Step 3.5 频域重组
        # ===============================================
        
        # Concat: [B, M, C] + [B, M, C] -> [B, M, 2C]
        f_concat = torch.cat([f_low_spatial, f_high_refined], dim=-1)
        
        # Linear Project: [B, M, 2C] -> [B, M, C]
        f_out = self.fusion_linear(f_concat)
        
        return f_out



