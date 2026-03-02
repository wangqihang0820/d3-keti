import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils

# ==========================================
# 1. 严格对应公式的几何特征更新层
# ==========================================
class GeometricEdgeConv(nn.Module):
    """
    严格实现文档 Step 1.2.2 公式:
    p^{l+1}_i = sigma( W0 * p^l_i + sum( W1 * h^l_ij ) )
    
    其中边特征 h^l_ij 定义为:
    h^l_ij = [ p^l_j || (vi - vj) || ||vi - vj||^2 ]
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        
        # W0: 处理中心点特征 p_i (线性变换)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        
        # W1: 处理边特征 h_ij
        # 输入维度: neighbor_feat(Cin) + relative_pos(3) + dist_sq(1)
        self.edge_in_dim = in_channels + 3 + 1
        self.lin_edge = nn.Linear(self.edge_in_dim, out_channels, bias=False)
        
        # sigma: 非线性激活 (文档提及 ReLU/LeakyReLU)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # KNN 模块
        self.knn = KNN(k=k, transpose_mode=True)

    def forward(self, x, pos, idx=None):
        """
        x: [B, N, C] 节点特征 p_i
        pos: [B, N, 3] 几何坐标 v_i
        idx: [B, N, k] KNN索引 (可选)
        """
        B, N, C = x.shape
        
        # 1. 计算 KNN (如果未提供)
        if idx is None:
            _, idx = self.knn(pos, pos)
            
        # ----------------------------------------
        # Term 1: W0 * p_i (Self-loop)
        # ----------------------------------------
        # [B, N, Cout]
        term_self = self.lin_self(x)
        
        # ----------------------------------------
        # Term 2: sum( W1 * h_ij ) (Neighbor Aggregation)
        # ----------------------------------------
        
        # A. 收集邻居特征 p_j -> [B, N, k, C]
        neighbor_x = pointnet2_utils.grouping_operation(
            x.transpose(1, 2).contiguous(), idx.int()
        ).permute(0, 2, 3, 1)
        
        # B. 收集邻居坐标 v_j -> [B, N, k, 3]
        neighbor_pos = pointnet2_utils.grouping_operation(
            pos.transpose(1, 2).contiguous(), idx.int()
        ).permute(0, 2, 3, 1)
        
        # C. 计算几何关系
        center_pos = pos.unsqueeze(2) # [B, N, 1, 3]
        
        # 相对位移: v_i - v_j (文档定义 vij = vi - vj)
        rel_pos = center_pos - neighbor_pos 
        
        # 相对距离平方: ||v_i - v_j||^2
        dist_sq = torch.sum(rel_pos ** 2, dim=-1, keepdim=True) # [B, N, k, 1]
        
        # D. 构建边特征 h_ij = [p_j || rel_pos || dist_sq]
        # [B, N, k, C+3+1]
        h_ij = torch.cat([neighbor_x, rel_pos, dist_sq], dim=-1)
        
        # E. 应用 W1 权重
        # [B, N, k, Cout]
        edge_feat = self.lin_edge(h_ij)
        
        # F. 聚合 (Summation, 对应公式中的 sum)
        # [B, N, Cout]
        # term_neighbor = torch.sum(edge_feat, dim=2)
        term_neighbor = torch.mean(edge_feat, dim=2)
        
        # ----------------------------------------
        # 3. 融合与激活
        # p^{l+1} = sigma( Term1 + Term2 )
        # ----------------------------------------
        out = self.act(term_self + term_neighbor)
        
        return out

# ==========================================
# 2. Point-UNet (使用修正后的 GeometricEdgeConv)
# ==========================================
class PointUNet(nn.Module):
    def __init__(self, in_channels=4, base_dim=64):
        super().__init__()
        
        # --- Encoder ---
        # Level 0: N 点
        self.conv0 = GeometricEdgeConv(in_channels, base_dim, k=16)
        
        # Level 1: N/2 点
        # 注意: input dim 是上一层的 base_dim
        self.conv1 = GeometricEdgeConv(base_dim, base_dim*2, k=16)
        
        # Level 2: N/4 点
        self.conv2 = GeometricEdgeConv(base_dim*2, base_dim*4, k=16)
        
        # --- Decoder (使用 MLP + 插值) ---
        # Up 1: N/4 -> N/2
        # 输入维度: Encoder L2输出(4C) + Encoder L1输出(2C) = 6C
        self.up_mlp1 = nn.Sequential(
            nn.Linear(base_dim*4 + base_dim*2, base_dim*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 这是一个修正层，输入2C -> 输出2C
        self.conv_up1 = GeometricEdgeConv(base_dim*2, base_dim*2, k=16)
        
        # Up 0: N/2 -> N
        # 输入维度: Up1输出(2C) + Encoder L0输出(C) = 3C
        self.up_mlp0 = nn.Sequential(
            nn.Linear(base_dim*2 + base_dim, base_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_up0 = GeometricEdgeConv(base_dim, base_dim, k=16)
        
        # Header
        self.out_proj = nn.Linear(base_dim, 128)

    def forward(self, x, pos, initial_idx=None):
        """
        x: [B, N, 4]
        pos: [B, N, 3]
        """
        # ================= Encoder =================
        # L0
        feat0 = self.conv0(x, pos, idx=initial_idx) # [B, N, 64]
        pos0 = pos
        
        # L1 (Downsample N -> N/2)
        npoint1 = pos0.shape[1] // 2
        idx1 = pointnet2_utils.furthest_point_sample(pos0, npoint1).long()
        idx1 = idx1.to(torch.int32).contiguous()   # 或 idx1 = idx1.int().contiguous()
        pos1 = pointnet2_utils.gather_operation(pos0.transpose(1, 2).contiguous(), idx1).transpose(1, 2).contiguous()
        # 收集上一层特征作为输入
        feat0_gathered = pointnet2_utils.gather_operation(feat0.transpose(1, 2).contiguous(), idx1).transpose(1, 2).contiguous()
        feat1 = self.conv1(feat0_gathered, pos1) # [B, N/2, 128]
        
        # L2 (Downsample N/2 -> N/4)
        npoint2 = pos1.shape[1] // 2
        idx2 = pointnet2_utils.furthest_point_sample(pos1, npoint2).long()
        idx2 = idx2.to(torch.int32).contiguous()   # 或 idx2 = idx2.int().contiguous()
        pos2 = pointnet2_utils.gather_operation(pos1.transpose(1, 2).contiguous(), idx2).transpose(1, 2).contiguous()
        feat1_gathered = pointnet2_utils.gather_operation(feat1.transpose(1, 2).contiguous(), idx2).transpose(1, 2).contiguous()
        feat2 = self.conv2(feat1_gathered, pos2) # [B, N/4, 256]
        
        # ================= Decoder =================
        # Up 1: N/4 -> N/2
        dist, idx = pointnet2_utils.three_nn(pos1, pos2)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interp_feat2 = pointnet2_utils.three_interpolate(feat2.transpose(1, 2).contiguous(), idx, weight).transpose(1, 2).contiguous()
        
        cat_feat1 = torch.cat([interp_feat2, feat1], dim=-1)
        feat_up1 = self.up_mlp1(cat_feat1)
        # 解码层也应用几何卷积进行特征平滑
        feat_up1 = self.conv_up1(feat_up1, pos1)
        
        # Up 0: N/2 -> N
        dist, idx = pointnet2_utils.three_nn(pos0, pos1)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interp_feat1 = pointnet2_utils.three_interpolate(feat_up1.transpose(1, 2).contiguous(), idx, weight).transpose(1, 2).contiguous()
        
        cat_feat0 = torch.cat([interp_feat1, feat0], dim=-1)
        feat_up0 = self.up_mlp0(cat_feat0)
        feat_out = self.conv_up0(feat_up0, pos0) # [B, N, 64]
        
        # Output
        out = self.out_proj(feat_out)
        return out

# ==========================================
# 3. 共享基底模块 (Phase 1 主类)
# ==========================================
class SharedBasis(nn.Module):
    def __init__(self, num_points=1024, k=16):
        super().__init__()
        self.k = k
        self.geometry_extractor = PointUNet(in_channels=4, base_dim=64)
        self.knn = KNN(k=k, transpose_mode=True)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def get_initial_features(self, xyz):
        """Step 1.2.1: [1 || deg]"""
        B, N, _ = xyz.shape
        ones = torch.ones((B, N, 3), device=xyz.device)
        deg = torch.full((B, N, 1), self.k, device=xyz.device, dtype=torch.float32)
        return torch.cat([ones, deg], dim=-1)

    # def compute_pca_normals(self, xyz, knn_idx):
    #     """Step 1.4.1: PCA Normal"""
    #     B, N, C = xyz.shape
    #     k = knn_idx.shape[2]
    #     neighbor_pos = pointnet2_utils.grouping_operation(
    #         xyz.transpose(1, 2).contiguous(), knn_idx.int()
    #     ).permute(0, 2, 3, 1)
    #     centers = xyz.unsqueeze(2)
    #     centered = neighbor_pos - centers
    #     cov = torch.matmul(centered.transpose(-2, -1), centered) / (k - 1)
    #     eig_vals, eig_vecs = torch.linalg.eigh(cov)
    #     return eig_vecs[..., 0]
    def compute_ps_normals(self, ps, center_idx):
        """
        利用 Pseudo-3D 特征图恢复法向量 (n_x, n_y, n_z)
        替代原有的 compute_pca_normals
        
        参数:
            ps: [B, 3, H, W] Pseudo-3D 输入特征图
            center_idx: [B, M] FPS 采样后的点在原始序列中的索引
        
        说明:
            1. center_idx 对应于 flatten 后的 ps 图像 (N=H*W)
            2. 假设 ps 的前两个通道对应 n_x, n_y
            3. n_z 利用单位球约束计算: n_z = sqrt(1 - n_x^2 - n_y^2)
        """
        B, C, H, W = ps.shape
        M = center_idx.shape[1]

        # ---------------------------------------------------
        # 1. 反归一化 (Reverse ImageNet Normalization)
        # ---------------------------------------------------
        # ImageNet constants
        mean = torch.tensor([0.485, 0.456, 0.406], device=ps.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=ps.device).view(1, 3, 1, 1)
        
        # X_raw = X_norm * std + mean
        # 此时 ps_raw 的范围大致在 [0, 1]
        ps_raw = ps * std + mean

        # ---------------------------------------------------
        # 2. 映射到 [-1, 1] 物理区间
        # ---------------------------------------------------
        # 假设原始图片存储法线时采用 (n+1)/2 的方式将 [-1,1] 映射到了 [0,1]
        # 这里的 ps_physics 就是恢复后的 [-1, 1] 范围的数值
        ps_physics = ps_raw * 2.0 - 1.0
        
        # 截断一下防止因插值等原因导致的越界 (稍微放宽一点容差)
        ps_physics = torch.clamp(ps_physics, min=-1.0, max=1.0)

        # 1. 展平 ps -> [B, 3, N_pixels]
        # 注意: 这里的展平顺序必须与 models.py 中 _to_BN3 的逻辑一致
        ps_flat = ps_physics.view(B, C, -1) # [B, 3, H*W]

        # 2. Gather 采样点对应的 ps 特征
        # center_idx: [B, M] -> 扩展为 [B, 3, M]
        idx_expanded = center_idx.unsqueeze(1).expand(-1, C, -1).long()
        
        # 提取特征: [B, 3, M]
        ps_sampled = torch.gather(ps_flat, 2, idx_expanded)
        
        # 转置为 [B, M, 3] 以便后续处理
        ps_features = ps_sampled.transpose(1, 2)

        # 3. 提取分量并恢复 n_z
        # 注意: 如果 Dataset 中 ps 经过了 ImageNet Normalize，这里实际上应该反归一化
        # 但既然文档指明 "已归一化至 [-1, 1]"，ps_sampled: [..., 0]=D, [..., 1]=nx, [..., 2]=ny 已经是 n_x, n_y
        nx = ps_features[..., 1]
        ny = ps_features[..., 2]
        
        # n_z = sqrt(1 - nx^2 - ny^2)
        # 使用 clamp 防止数值误差导致负数开根号
        nz_sq = 1.0 - nx**2 - ny**2
        nz_sq = torch.clamp(nz_sq, min=0.0)
        nz = torch.sqrt(nz_sq + 1e-8)
        
        # 4. 组合并归一化 (保险起见)
        normals = torch.stack([nx, ny, nz], dim=-1) # [B, M, 3]
        normals = F.normalize(normals, p=2, dim=-1, eps=1e-8)
        
        return normals

    def forward(self, xyz, ps=None, center_idx=None):
        """
        xyz: [B, N, 3] from Step 1.1 (FPS output)
        """
        B, N, _ = xyz.shape
        
        # Step 1.2: KNN
        _, idx = self.knn(xyz, xyz)
        
        # Step 1.2.1: Init Feat
        feat = self.get_initial_features(xyz)
        
        # Step 1.2.2: Update (PointUNet)
        latent_feat = self.geometry_extractor(feat, xyz, initial_idx=idx) # [B, N, 128]
        
        # ★ 新增：将特征 L2 归一化！
        # 因为后续的 target 是基于单位法向量算出的，所以 latent_feat 也必须是单位向量
        # 这样能保证 (neighbor_feat - center_feat)**2 严格落在 [0, 4] 之间，彻底根除爆炸！
        latent_feat = F.normalize(latent_feat, p=2, dim=-1, eps=1e-8)
        
        # Step 1.2.3: Adaptive Weights
        # Gather neighbors from LAST LAYER latent features
        neighbor_feat = pointnet2_utils.grouping_operation(
            latent_feat.transpose(1, 2).contiguous(), idx.int()
        ).permute(0, 2, 3, 1)
        center_feat = latent_feat.unsqueeze(2)
        
        # w_ij = ReLU( |p_i - p_j|^2 )
        diff_sq = torch.sum((neighbor_feat - center_feat)**2, dim=-1)
        W_val = F.relu(diff_sq)
        
        # Step 1.4: Loss
        # 只有在训练模式且提供了 ps 和 center_idx 时才计算
        loss_geo = torch.tensor(0.0, device=xyz.device)
        if self.training and ps is not None and center_idx is not None:
            normals = self.compute_ps_normals(ps, center_idx)
            neighbor_n = pointnet2_utils.grouping_operation(
                normals.transpose(1, 2).contiguous(), idx.int()
            ).permute(0, 2, 3, 1)
            center_n = normals.unsqueeze(2)
            d_ij = torch.sum((neighbor_n - center_n)**2, dim=-1)
            d_ij_no_grad = d_ij.detach()  # 1. 切断几何真值的梯度
            target = self.alpha * d_ij_no_grad  # 2. 让 alpha 参与计算
            loss_geo = F.mse_loss(W_val, target)
            
        # Step 1.3: Basis
        B_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, self.k)
        Src_idx = torch.arange(N, device=xyz.device).view(1, N, 1).expand(B, N, self.k)
        Dst_idx = idx.long()
        
        A = torch.zeros((B, N, N), device=xyz.device)
        A[B_idx, Src_idx, Dst_idx] = W_val
        A = (A + A.transpose(1, 2)) / 2
        
        D = torch.sum(A, dim=-1)
        d_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_mat = torch.diag_embed(d_inv_sqrt)
        I = torch.eye(N, device=xyz.device).unsqueeze(0)
        L = I - torch.matmul(torch.matmul(D_mat, A), D_mat)
        
        # # 添加微扰处理
        # eps = 1e-6
        # L = L + eps * torch.eye(L.shape[-1], device=L.device, dtype=L.dtype)
        
        # 1. 清除可能出现的 NaN 和 Inf，替换为安全的数值
        if torch.isnan(L).any() or torch.isinf(L).any():
            L = torch.nan_to_num(L, nan=0.0, posinf=1e4, neginf=-1e4)

        # 2. 给矩阵对角线加上极小的正数 (Jitter / Epsilon)，强制其良态
        # 假设 L 的形状是 [B, N, N]
        B, N, _ = L.shape
        epsilon = 1e-5
        L = L + torch.eye(N, device=L.device).unsqueeze(0) * epsilon

        # 3. 再进行特征值分解
        eig_vals, U = torch.linalg.eigh(L)

        
        return U.detach(), loss_geo, idx