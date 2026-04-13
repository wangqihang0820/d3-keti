# 基于 PEFT-SD 的逻辑
import torch
import torch.nn as nn
from knn_cuda import KNN

class SharedBasis(nn.Module):
    """
    基于 PEFT-SD 逻辑优化的无参数全局共享基底计算模块
    通过纯空间坐标的距离度量直接构建图拉普拉斯矩阵
    """
    def __init__(self, num_points=4096, k=16):
        super().__init__()
        # 空间分支 (Spatial Branch) 仍然需要 k-NN 索引，所以保留该组件
        self.knn = KNN(k=k, transpose_mode=True)

    def get_laplacian(self, adj_matrix, normalize=True):
        """
        计算图的拉普拉斯矩阵 (Graph Laplacian)
        """
        if normalize:
            # 度矩阵 (Degree matrix): 按行求和
            D = torch.sum(adj_matrix, dim=-1)  # (B, N)
            # 防止除零错误，加上 epsilon
            D_inv_sqrt = torch.rsqrt(D + 1e-6)  # (B, N)
            D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # (B, N, N)
            # 归一化拉普拉斯矩阵: L = I - D^{-1/2} A D^{-1/2}
            I = torch.eye(adj_matrix.size(-1), device=adj_matrix.device).unsqueeze(0)
            L = I - torch.bmm(torch.bmm(D_inv_sqrt, adj_matrix), D_inv_sqrt)
        else:
            D = torch.sum(adj_matrix, dim=-1)
            D = torch.diag_embed(D)
            L = D - adj_matrix
            
        return L

    def get_basis(self, center):
        """
        根据物理坐标计算全局正交基底 U
        center: [B, N, 3] 物理坐标
        """
        # 1. 计算所有点对之间的欧式距离矩阵 [B, N, N]
        dist_matrix = torch.cdist(center, center)
        
        # 2. 将距离转化为相似度/邻接矩阵
        # 距离越近，相似度越趋近于 1；距离越远，相似度越小
        # 加上 identity matrix 防止除零，并强化 self-loop
        min_dist = torch.min(dist_matrix[dist_matrix > 0], dim=-1, keepdim=True).values
        # 注意防爆：如果 min_dist 出现极端情况为空，给个保护值
        if min_dist.numel() == 0:
            min_dist = torch.tensor([1e-6], device=center.device)
            
        I = torch.eye(dist_matrix.size(-1), device=dist_matrix.device).unsqueeze(0)
        adj_matrix = 1.0 / ((dist_matrix / min_dist) + I)
        
        # 3. 计算归一化拉普拉斯矩阵
        L = self.get_laplacian(adj_matrix, normalize=True)
        
        # 为了极度安全，给对角线加上极小微扰，确保数值稳定和正定
        L = L + I * 1e-6
        
        # 4. 特征值分解获取基底 U
        # eig_vals, U = torch.linalg.eigh(L)
        _, U = torch.linalg.eigh(L)
        
        # 保留 PEFT-SD 的设定，不进行转置 (保持与 GFT 乘法方向对应)
        return U

    def forward(self, xyz, ps=None, center_idx=None):
        """
        Args:
            xyz: [B, N, 3] 输入的前景采样坐标 (4096个点)
            ps:  保留接口兼容
            center_idx: 保留接口兼容
        Returns:
            U: [B, N, N] 正交基底 (需 detach 切断梯度，作为纯数学先验)
            loss_geo: 0.0 (因为没有参数网络了，几何损失废弃)
            knn_idx: [B, N, k] 供空间分支使用的邻居索引
        """
        # 1. 直接通过 PEFT-SD 逻辑计算全局基底
        U = self.get_basis(xyz)
        
        # 2. 为空间互补分支 (Spatial Branch) 计算 knn 索引
        _, knn_idx = self.knn(xyz, xyz)
        
        # 3. 兼容旧接口，返回 0.0 的几何损失
        # loss_geo = torch.tensor(0.0, device=xyz.device, requires_grad=True)
        loss_geo = torch.tensor(0.0, device=xyz.device)
        
        # 必须返回 U.detach()，确保基底分解操作不参与反向传播，避免内存泄漏和 NaN
        return U.detach(), loss_geo, knn_idx
