import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareOT(nn.Module):
    """
    不确定性感知最优传输模块 (Uncertainty-Aware OT)
    训练专用：实现 "误差大 -> 权重小" 的自适应分配，平衡多模态训练。
    """
    def __init__(self, momentum=0.9, init_temperature=0.2, tau=1.0):
        super().__init__()
        self.momentum = momentum
        self.tau = tau
        
        # 可学习的温度系数，控制分配的“挑剔”程度
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        
        # 注册缓冲区：保存 2D/3D 误差的全局滑动平均值
        self.register_buffer('mu_2d', torch.tensor(1.0))
        self.register_buffer('mu_3d', torch.tensor(1.0))
        self.register_buffer('is_initialized', torch.tensor(False))

    def _update_stats(self, l2d_detach, l3d_detach):
        if self.training:
            batch_mean_2d = l2d_detach.mean()
            batch_mean_3d = l3d_detach.mean()
            
            if not self.is_initialized:
                self.mu_2d.data.copy_(batch_mean_2d)
                self.mu_3d.data.copy_(batch_mean_3d)
                self.is_initialized.fill_(True)
            else:
                self.mu_2d.data.mul_(self.momentum).add_(batch_mean_2d * (1 - self.momentum))
                self.mu_3d.data.mul_(self.momentum).add_(batch_mean_3d * (1 - self.momentum))

    def forward(self, cost_2d, cost_3d):
        cost_2d = cost_2d.view(-1)
        cost_3d = cost_3d.view(-1)
        
        # 1. 更新统计量
        self._update_stats(cost_2d.detach(), cost_3d.detach())
        
        # 2. 动态归一化 (无量纲化)
        c2d_norm = cost_2d / (self.mu_2d + 1e-6)
        c3d_norm = cost_3d / (self.mu_3d + 1e-6)
        
        # 3. 构建源分布 P0 (Trust Supply)
        # 误差相对越小，本金越多
        p0_logits = torch.stack([-c2d_norm / self.tau, -c3d_norm / self.tau], dim=1)
        p0 = F.softmax(p0_logits, dim=1) 
        
        # 4. 求解 OT (Softmin)
        temperature = F.softplus(self.log_temperature) + 1e-6
        cost_stack = torch.stack([c2d_norm, c3d_norm], dim=1)
        
        energy = -cost_stack / temperature
        unnormalized_w = p0 * torch.exp(energy)
        
        # 归一化
        weights = unnormalized_w / (unnormalized_w.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights[:, 0], weights[:, 1]