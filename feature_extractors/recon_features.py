import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.models import Model  
from utils.utils import KNNGaussianBlur
from feature_extractors.change_ex import CombinedExchange   
from models.cross_attention import BiDirectionalCrossAttention  
from models.lapwavegate import LapWaveGate
from utils.ot_fusion import fuse_by_task_ot,robust_zscore
from feature_extractors.ReconNet2D import ReconNet2D
from feature_extractors.ReconNet3D import ReconNet3D
from feature_extractors.shared_basis import SharedBasis  
from feature_extractors.masking import LatentRandomMasking
from feature_extractors.spectral_branch import *
from feature_extractors.spatial_branch import SpatialComplementBranch
from feature_extractors.fusion_spatial_spectral import DualStreamFusion
from feature_extractors.ot_module import UncertaintyAwareOT

# ------------------------------------------------
# 主干网络：2D / ps 编码 + CSS + 3D 编码 + Cross-Attn + 重建
# ------------------------------------------------
class FusionReconNet(nn.Module):
    def __init__(self,
                 rgb_backbone_name="vit_base_patch8_224_dino",
                 xyz_backbone_name="Point_MAE",
                 group_size=128,
                 num_group=1024,
                 img_size=224):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        self.encoder = Model(
            device=self.device,
            rgb_backbone_name=rgb_backbone_name,
            xyz_backbone_name=xyz_backbone_name,
            group_size=group_size,
            num_group=num_group,
        )

        self.shared_basis = SharedBasis(num_points=num_group, k=16)
        self.rgb_feat_dim = 768    
        self.rgb_feat_hw = 28 * 28  

        self.css = CombinedExchange(p=2)

        self.xyz_token_dim = 1152
        self.xyz_proj = nn.Linear(self.xyz_token_dim, self.rgb_feat_dim)
        
        self.xyz_feat_head = nn.Linear(self.rgb_feat_dim, self.xyz_token_dim)
        self.xyz_coord_head = nn.Linear(self.rgb_feat_dim, 3)
        
        self.masking_module = LatentRandomMasking(
            input_dim=self.rgb_feat_dim,
            mask_ratio=0.75               
        )

        self.spectral_transform = SpectralTransform()
        self.cbdg = DynamicContentGating(num_points=num_group)
        
        self.fd_moe = FD_MoE(channels=self.rgb_feat_dim, num_experts=4, top_k=2)
        self.ggrm = GGRM(channels=self.rgb_feat_dim, num_points=1024, reduction=4)
        self.spatial_branch = SpatialComplementBranch(channels=self.rgb_feat_dim)
        self.fusion = DualStreamFusion(channels=self.rgb_feat_dim)
        
        self.bdca = BiDirectionalCrossAttention(
            dim_2d=self.rgb_feat_dim,     
            dim_3d=self.rgb_feat_dim,     
            num_heads=8,
            drop=0.1,
            order="2d_first",
        )

        feat_size = img_size // 8 
        self.recon_2d = ReconNet2D(
            in_channels=self.rgb_feat_dim, 
            embed_dim=96,                  
            img_size=feat_size,            
            window_size=7                  
        )
        self.recon_3d = ReconNet3D(in_dim=self.rgb_feat_dim)
        
        self.register_buffer('err_2d_mean', torch.zeros(1, img_size, img_size))
        self.register_buffer('err_2d_std', torch.ones(1, img_size, img_size))
        self.register_buffer('err_3d_mean', torch.zeros(1, img_size, img_size))
        self.register_buffer('err_3d_std', torch.ones(1, img_size, img_size))
        
        self.register_buffer('depth_mean', torch.zeros(1, 1, img_size, img_size))
        self.register_buffer('depth_std', torch.ones(1, 1, img_size, img_size))
        
        for param in self.encoder.rgb_backbone.parameters():
            param.requires_grad = False
        # ★ 解冻 PointMAE！让它适应工业工件的 2.5D 平坦地形
        # for param in self.encoder.xyz_backbone.parameters():
        #     param.requires_grad = False
            
        self.encoder.rgb_backbone.eval()
        # 注意：因为解冻了，训练时可以进入 train 模式
        # self.encoder.xyz_backbone.eval()

    def forward(self, rgb, xyz, sampled_idx, ps):
        """
        xyz: [B, 4096, 3] (已纯净采样)
        sampled_idx: [B, 4096] (这4096个点在 50176 原图中的位置索引)
        """
        self.encoder.rgb_backbone.eval()
        
        B = rgb.size(0)

        rgb_feats = self.encoder.forward_rgb_features(rgb)  
        ps_feats  = self.encoder.forward_rgb_features(ps)   

        rgb_feats_css, ps_feats_css = self.css(rgb_feats, ps_feats)
        rgb_feats_bottleneck = rgb_feats_css.detach()

        # -------------------------
        # 3D 编码与 ★ 黄金索引映射
        # -------------------------
        # xyz 已经是完美的 [B, 4096, 3]，直接喂入！
        xyz_tokens, center, ori_idx, center_idx_internal = self.encoder.xyz_backbone(xyz)
        
        # ！！！黄金映射！！！
        # 将 PointMAE 内部返回的 0~4095 索引，精准投射回 0~50175 的图像全局像素索引
        # center_idx = torch.gather(sampled_idx, 1, center_idx_internal.long())
        # ==========================================
        # ★ 终极防爆网：防止极端样本产生 NaN，导致 .long() 转换出亿级错误索引！
        # ==========================================
        # 1. 过滤内部 4096 索引
        center_idx_internal = torch.nan_to_num(center_idx_internal, nan=0.0)
        center_idx_internal = torch.clamp(center_idx_internal, min=0, max=4095).long()
        
        # ！！！黄金映射！！！
        center_idx = torch.gather(sampled_idx, 1, center_idx_internal)
        
        # 2. 再次上保险，确保投射到 224x224 的全局索引绝对不越过 50175 的边界
        center_idx = torch.nan_to_num(center_idx, nan=0.0)
        center_idx = torch.clamp(center_idx, min=0, max=50175).long()

        U, loss_geo, knn_idx = self.shared_basis(center, ps, center_idx)
        
        xyz_tokens = xyz_tokens.transpose(1, 2).contiguous()          
        xyz_tokens = torch.nan_to_num(xyz_tokens, nan=0.0, posinf=1e4, neginf=-1e4)
        
        F_feat_gt = xyz_tokens.detach()   
        F_coord_gt = center.detach()      
        
        F_in_raw = self.xyz_proj(xyz_tokens.detach())

        if self.training:
            F_in, mask = self.masking_module(F_in_raw)
        else:
            F_in, mask = self.masking_module(F_in_raw)
            
        F_spec = self.spectral_transform.gft(F_in, U)  
        F_low, F_high, gate_map = self.cbdg(F_spec)
        F_low_spatial = self.fd_moe(F_low, U)
        xyz_features_processed = self.ggrm(F_low_spatial, F_high, U)
        F_spatial_out = self.spatial_branch(F_in, knn_idx)
        
        xyz_recon_final = self.fusion(
            f_freq_out=xyz_features_processed, 
            f_spatial_out=F_spatial_out, 
            f_in=F_in
        )
        
        B2, C_rgb, H_rgb, W_rgb = rgb_feats_bottleneck.shape
        N_rgb = H_rgb * W_rgb
        rgb_tokens = rgb_feats_bottleneck.view(B2, C_rgb, N_rgb).permute(0, 2, 1)

        rgb_updated, xyz_updated = self.bdca(rgb_tokens, xyz_recon_final)
        
        B, N, D = rgb_updated.shape          
        h = w = int(N ** 0.5)               
        assert h * w == N, (N, h, w)
        rgb_in_recon = rgb_updated.permute(0, 2, 1).contiguous().reshape(B, D, h, w)  
        
        rgb_recon_feat = self.recon_2d(rgb_in_recon)
        xyz_recon_feat = self.recon_3d(xyz_updated, center)
        
        xyz_recon_feat_out = self.xyz_feat_head(xyz_recon_feat)   
        xyz_recon_coord_out = self.xyz_coord_head(xyz_recon_feat) 
        
        return rgb_recon_feat, xyz_recon_feat_out, xyz_recon_coord_out, rgb_feats_css, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx


# ------------------------------------------------
# 辅助函数：精准散布到 2D
# ------------------------------------------------
def splat_3d_error_to_2d_exact(errors, center_idx, img_size):
    B, N = errors.shape
    error_map = torch.zeros((B, img_size * img_size), device=errors.device)
    
    valid_mask = (center_idx > 0).float()
    errors_clean = errors * valid_mask
    
    # 此时 center_idx 已经是 0~50175，直接 scatter_ 完美归位！
    error_map.scatter_(1, center_idx.long(), errors_clean)
    error_map = error_map.view(B, 1, img_size, img_size)
    
    # 3. ★ 峰值膨胀连通 (Dilation)
    error_map_dilated = F.max_pool2d(error_map, kernel_size=11, stride=1, padding=5)
    
    # 4. 轻柔化边缘，消除方块马赛克感
    error_map_smooth = F.avg_pool2d(error_map_dilated, kernel_size=5, stride=1, padding=2)
    
    # # 3. ★ 峰值膨胀连通 (Dilation) -> 改为 5x5，紧凑连通，绝不蔓延
    # error_map_dilated = F.max_pool2d(error_map, kernel_size=3, stride=1, padding=1)
    
    # # 4. 轻柔化边缘，消除方块马赛克感 -> 稍微加大平滑核 7x7，让图更像热力图
    # error_map_smooth = F.avg_pool2d(error_map_dilated, kernel_size=7, stride=1, padding=3)
    
    return error_map_smooth.squeeze(1)

# ------------------------------------------------
# 3. 重建式异常检测封装
# ------------------------------------------------
class ReconFeatures(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.img_size_val = getattr(args, 'img_size', 224)
        self.net = FusionReconNet(
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group,
            img_size=args.img_size_val,
        ).to(self.device)

        self.blur = KNNGaussianBlur(2)
        self.l2_weight = 0.5
        self.lambda_geo = getattr(args, 'lambda_geo', 0.1) 
        self.ot_module = UncertaintyAwareOT(momentum=0.9, init_temperature=0.2).to(self.device)
        self.reset_buffers()
        
        self.weight_2d = 1.0           
        self.weight_3d_feat = 2.0      
        self.weight_depth = 0.3        

    def reset_buffers(self):
        self.image_preds = []   
        self.image_labels = []  
        self.pixel_preds = []   
        self.pixel_labels = []  
        self.gts = []           
        self.pred_maps = []     
        self.vis_samples = []   
        self.maps_2d = [] 
        self.maps_3d = []

    def compute_hybrid_loss(self, pred, target, dim=-1, mask=None, use_smooth_l1=False, beta=0.1):
        pred_safe = pred + 1e-8
        target_safe = target + 1e-8
        
        cosine_loss = 1 - F.cosine_similarity(pred_safe, target_safe, dim=dim)
        
        pred_norm = F.normalize(pred, p=2, dim=dim)
        target_norm = F.normalize(target, p=2, dim=dim)
        
        # ==========================================
        # ★ 核心改动：高频宽容机制 (High-Frequency Tolerance)
        # ==========================================
        if use_smooth_l1:
            # 误差 > beta 时转换为线性惩罚，防止边缘的高频拓扑抖动引发梯度爆炸
            l2_loss_raw = F.smooth_l1_loss(pred_norm, target_norm, reduction='none', beta=beta)
        else:
            # 原始的 MSE，对微小误差进行平方放大
            l2_loss_raw = F.mse_loss(pred_norm, target_norm, reduction='none')
        
        if dim == 1: 
            l2_loss = l2_loss_raw.mean(dim=1)
        else:        
            l2_loss = l2_loss_raw.mean(dim=-1)
            
        total_loss_map = cosine_loss + self.l2_weight * l2_loss
        
        if mask is not None:
            total_loss_map = total_loss_map * mask
            return total_loss_map.sum() / (mask.sum() + 1e-5)
        else:
            return total_loss_map.mean()

    def train_step(self, sample):
        # ★ 接收 5 个元素，取出 sampled_idx
        rgb, xyz, sampled_idx, depth_map, ps = sample

        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        sampled_idx = sampled_idx.to(self.device)
        ps = ps.to(self.device)

        rgb_recon, xyz_feat_recon, _, rgb_target, F_feat_gt, _, loss_geo, U, mask, center, center_idx = self.net(rgb, xyz, sampled_idx, ps)

        depth_for_mask = depth_map.to(self.device)
        if depth_for_mask.dim() == 4:
            if depth_for_mask.shape[1] == 3: depth_for_mask = depth_for_mask[:, 2, :, :]
            elif depth_for_mask.shape[-1] == 3: depth_for_mask = depth_for_mask[:, :, :, 2]
            else: depth_for_mask = depth_for_mask.mean(dim=1)
        if depth_for_mask.dim() == 3:
            depth_for_mask = depth_for_mask.unsqueeze(1)
            
        batch_min_z = depth_for_mask.view(rgb.size(0), -1).min(dim=1)[0].view(rgb.size(0), 1, 1, 1)
        fg_mask_224 = (depth_for_mask > batch_min_z + 1e-5).float()
        
        bg_mask = 1.0 - fg_mask_224
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=15, stride=1, padding=7)
        train_mask_224 = 1.0 - eroded_bg
        train_mask_28 = F.interpolate(train_mask_224, size=(28, 28), mode='nearest').squeeze(1) 

        # 2D 保持原有的 MSE，维持对 RGB 纹理细节的极高敏感度
        loss_2d = self.compute_hybrid_loss(
            rgb_recon, rgb_target.detach(), dim=1, mask=train_mask_28, 
            use_smooth_l1=False
        ) 
        
        effective_3d_mask = mask
        # ==========================================
        # ★ 启用 3D 边界宽容
        # 设定 beta=0.1：当重建误差 < 0.1（如平坦表面的真实微小缺陷）时，依旧使用 L2 进行精细优化；
        # 当重建误差 > 0.1（如把手边缘因采样和插值导致的剧烈拓扑畸变）时，转为 L1 线性惩罚，不让模型过度纠结于修复边缘。
        # ==========================================
        loss_3d_feat = self.compute_hybrid_loss(
            xyz_feat_recon, F_feat_gt, dim=2, mask=effective_3d_mask, 
            use_smooth_l1=True, beta=0.1
        )
        loss_3d = loss_3d_feat 
        
        w_alpha = 1.0
        w_beta = 10.0 # 根据你之前选择的 50 进行保留
    
        weighted_loss = w_alpha * loss_2d + w_beta * loss_3d
        total_loss = weighted_loss + self.lambda_geo * loss_geo
        
        return {
            "loss": total_loss,
            "l2d": loss_2d.item(),
            "l3d": loss_3d.item(),
            "geo": loss_geo.item(),
            "alpha": w_alpha,
            "beta": w_beta
        }
        
    @torch.no_grad()
    def build_error_statistics(self, train_loader):
        self.net.eval()
        err_2d_list = []
        err_3d_feat_list = []
        depth_list = []
        
        from tqdm import tqdm
        print("\n[Post-Training] Building Pixel-wise Z-Score Baseline...")
        for sample, _ in tqdm(train_loader, desc="Extracting Baseline"):
            # ★
            rgb, xyz, sampled_idx, depth_map, ps = sample
            rgb, xyz, sampled_idx, ps = rgb.to(self.device), xyz.to(self.device), sampled_idx.to(self.device), ps.to(self.device)
            B = rgb.size(0)

            ensemble_runs = 4  
            num_points = 1024
            
            err_3d_feat_ensemble = torch.zeros(B, num_points, device=self.device)
            err_2d_ensemble = torch.zeros(B, self.img_size_val, self.img_size_val, device=self.device)
            mask_visit_count = torch.zeros(B, num_points, device=self.device) + 1e-5

            rand_indices = torch.randperm(num_points, device=self.device).expand(B, -1)
            chunk_size = num_points // ensemble_runs
            
            for run_idx in range(ensemble_runs):
                current_visible_idx = rand_indices[:, run_idx*chunk_size : (run_idx+1)*chunk_size]
                current_mask = torch.ones(B, num_points, device=self.device)
                current_mask.scatter_(1, current_visible_idx, 0.0)
                self.net.masking_module.fixed_mask = current_mask

                rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx = self.net(rgb, xyz, sampled_idx, ps)
                
                xyz_feat_norm = F.normalize(xyz_feat_recon, p=2, dim=2)
                F_feat_gt_norm = F.normalize(F_feat_gt, p=2, dim=2)
                # err_3d_feat_current = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                #                       self.l2_weight * torch.mean((xyz_feat_norm - F_feat_gt_norm) ** 2, dim=2)
                # ★ 新代码：保持与训练阶段相同的 Smooth L1 宽容度
                l2_diff_3d = F.smooth_l1_loss(xyz_feat_norm, F_feat_gt_norm, reduction='none', beta=0.1).mean(dim=2)
                err_3d_feat_current = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                                    self.l2_weight * l2_diff_3d
                err_3d_feat_ensemble += (err_3d_feat_current * mask)
                mask_visit_count += mask

                rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)
                rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
                err_2d_current = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                                 self.l2_weight * torch.mean((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
                if err_2d_current.shape[-1] != self.img_size_val:
                    err_2d_current = F.interpolate(err_2d_current.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)
                err_2d_ensemble += err_2d_current

            self.net.masking_module.fixed_mask = None

            err_3d_total = err_3d_feat_ensemble / mask_visit_count
            err_2d_total = err_2d_ensemble / ensemble_runs
            
            err_3d_map = splat_3d_error_to_2d_exact(err_3d_total, center_idx, self.img_size_val)

            depth_gt = depth_map.to(self.device)
            if depth_gt.dim() == 4:
                if depth_gt.shape[1] == 3: depth_gt = depth_gt[:, 2:3, :, :]
                elif depth_gt.shape[-1] == 3: depth_gt = depth_gt[:, :, :, 2:3]
                else: depth_gt = depth_gt.mean(dim=1, keepdim=True)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)
                
            err_2d_smooth = self.blur(err_2d_total.unsqueeze(1).cpu()).squeeze(1)
            err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).squeeze(1)

            err_2d_list.append(err_2d_smooth)
            err_3d_feat_list.append(err_3d_smooth)
            depth_list.append(depth_gt.cpu())
            
        all_err_2d = torch.cat(err_2d_list, dim=0) 
        all_err_3d_feat = torch.cat(err_3d_feat_list, dim=0)
        all_depth = torch.cat(depth_list, dim=0)

        self.net.err_2d_mean.copy_(all_err_2d.mean(dim=0, keepdim=True))
        self.net.err_2d_std.copy_(all_err_2d.std(dim=0, keepdim=True) + 1e-5)
        self.net.err_3d_mean.copy_(all_err_3d_feat.mean(dim=0, keepdim=True))
        self.net.err_3d_std.copy_(all_err_3d_feat.std(dim=0, keepdim=True) + 1e-5)
        self.net.depth_mean.copy_(all_depth.mean(dim=0, keepdim=True))
        self.net.depth_std.copy_(all_depth.std(dim=0, keepdim=True) + 1e-5)
        print("Baseline and Physical Depth Template established and locked!")

    @torch.no_grad()
    def predict(self, sample, gt, label, rgb_path=None):
        if not isinstance(sample, (tuple, list)):
            raise RuntimeError("ReconFeatures.predict 期望 sample 是 tuple/list")

        # ★
        rgb, xyz, sampled_idx, depth_map, ps = sample

        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        sampled_idx = sampled_idx.to(self.device)
        depth_map = depth_map.to(self.device)
        ps = ps.to(self.device)
        B = rgb.size(0)
        
        ensemble_runs = 4  
        num_points = 1024

        err_3d_feat_ensemble = torch.zeros(B, num_points, device=self.device)
        err_2d_ensemble = torch.zeros(B, self.img_size_val, self.img_size_val, device=self.device)
        mask_visit_count = torch.zeros(B, num_points, device=self.device) + 1e-5

        rand_indices = torch.randperm(num_points, device=self.device).expand(B, -1)
        chunk_size = num_points // ensemble_runs

        for run_idx in range(ensemble_runs):
            current_visible_idx = rand_indices[:, run_idx*chunk_size : (run_idx+1)*chunk_size]
            current_mask = torch.ones(B, num_points, device=self.device)
            current_mask.scatter_(1, current_visible_idx, 0.0)
            self.net.masking_module.fixed_mask = current_mask

            rgb_recon, xyz_feat_recon, xyz_recon_coord_out, rgb_target, F_feat_gt, F_coord_gt, loss_geo, U, mask, center, center_idx = self.net(rgb, xyz, sampled_idx, ps)
            
            xyz_feat_norm = F.normalize(xyz_feat_recon, p=2, dim=2)
            F_feat_gt_norm = F.normalize(F_feat_gt, p=2, dim=2)
            # err_3d_feat_current = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
            #                       self.l2_weight * torch.mean((xyz_feat_norm - F_feat_gt_norm) ** 2, dim=2)
            # ★ 新代码：推理时同步使用 Smooth L1 防止边界得分爆炸
            l2_diff_3d = F.smooth_l1_loss(xyz_feat_norm, F_feat_gt_norm, reduction='none', beta=0.1).mean(dim=2)
            err_3d_feat_current = (1 - F.cosine_similarity(xyz_feat_recon + 1e-8, F_feat_gt + 1e-8, dim=2)) + \
                                self.l2_weight * l2_diff_3d
            err_3d_feat_ensemble += (err_3d_feat_current * mask)
            mask_visit_count += mask

            rgb_recon_norm = F.normalize(rgb_recon, p=2, dim=1)   
            rgb_target_norm = F.normalize(rgb_target, p=2, dim=1)
            err_2d_current = (1 - F.cosine_similarity(rgb_recon + 1e-8, rgb_target + 1e-8, dim=1)) + \
                             self.l2_weight * torch.mean((rgb_recon_norm - rgb_target_norm) ** 2, dim=1)
            if err_2d_current.shape[-1] != self.img_size_val:
                err_2d_current = F.interpolate(err_2d_current.unsqueeze(1), size=(self.img_size_val, self.img_size_val), mode='bilinear', align_corners=False).squeeze(1)
            err_2d_ensemble += err_2d_current

        self.net.masking_module.fixed_mask = None 

        err_3d_total = err_3d_feat_ensemble / mask_visit_count
        err_2d_total = err_2d_ensemble / ensemble_runs

        err_3d_map = splat_3d_error_to_2d_exact(err_3d_total, center_idx, self.img_size_val)

        depth_for_eval = depth_map.to(self.device)
        if depth_for_eval.dim() == 4:
            if depth_for_eval.shape[1] == 3: depth_for_eval = depth_for_eval[:, 2:3, :, :]
            elif depth_for_eval.shape[-1] == 3: depth_for_eval = depth_for_eval[:, :, :, 2:3]
            else: depth_for_eval = depth_for_eval.mean(dim=1, keepdim=True)
        if depth_for_eval.dim() == 3:
            depth_for_eval = depth_for_eval.unsqueeze(1)
            
        batch_min_z = depth_for_eval.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        fg_mask_raw = (depth_for_eval > batch_min_z + 1e-5).float()
        
        err_2d_smooth = self.blur(err_2d_total.unsqueeze(1).cpu()).to(self.device)
        err_2d_smooth = err_2d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        err_3d_smooth = self.blur(err_3d_map.unsqueeze(1).cpu()).to(self.device)
        err_3d_smooth = err_3d_smooth.view(B, self.img_size_val, self.img_size_val)
        
        # 1最原始的
        # std_2d = self.net.err_2d_std.to(self.device)
        # std_3d_feat = self.net.err_3d_std.to(self.device)
        
        # eps_2d = torch.clamp(std_2d.mean() * 0.1, min=1e-3)
        # eps_3d = torch.clamp(std_3d_feat.mean() * 0.1, min=1e-3)
        
        # err_2d_norm = F.relu(err_2d_smooth - self.net.err_2d_mean.to(self.device)) / (std_2d + eps_2d)
        # err_3d_feat_norm = F.relu(err_3d_smooth - self.net.err_3d_mean.to(self.device)) / (std_3d_feat + eps_3d)
        
        # std_depth = self.net.depth_std.to(self.device)
        # eps_depth = torch.clamp(std_depth.mean() * 0.1, min=1e-3)
        
        # err_depth_raw = torch.abs(depth_for_eval - self.net.depth_mean.to(self.device))
        # err_depth_norm = err_depth_raw.squeeze(1) / (std_depth.squeeze(1) + eps_depth)


        # 2================== ★ 终极修正：形态学防抖基线 (Morphological Baseline) ==================
        # mean_2d = self.net.err_2d_mean.to(self.device)
        # std_2d = self.net.err_2d_std.to(self.device)
        
        # mean_3d_raw = self.net.err_3d_mean.to(self.device)
        # std_3d_raw = self.net.err_3d_std.to(self.device)

        # # 对 3D 训练集的“均值”和“标准差”同时使用 7x7 进行膨胀。
        # # 这意味着允许测试图在各个方向上有 3 个像素的空间错位，彻底消灭“月牙形错位红斑”！
        # mean_3d_tolerant = F.max_pool2d(mean_3d_raw, kernel_size=51, stride=1, padding=25)
        # std_3d_tolerant = F.max_pool2d(std_3d_raw, kernel_size=51, stride=1, padding=25)
        
        # eps_2d = torch.clamp(std_2d.mean() * 0.1, min=1e-3)
        # eps_3d = torch.clamp(std_3d_tolerant.mean() * 0.1, min=1e-3)
        
        # # 2D 保持极高的敏感度不变
        # err_2d_norm = F.relu(err_2d_smooth - mean_2d) / (std_2d + eps_2d)
        
        # # 3D 减去带有“防抖罩”的均值，真实的缺陷依然会刺透它！
        # err_3d_feat_norm = F.relu(err_3d_smooth - mean_3d_tolerant) / (std_3d_tolerant + eps_3d)


        # std_depth = self.net.depth_std.to(self.device)
        # eps_depth = torch.clamp(std_depth.mean() * 0.1, min=1e-3)
        
        # err_depth_raw = torch.abs(depth_for_eval - self.net.depth_mean.to(self.device))
        # err_depth_norm = err_depth_raw.squeeze(1) / (std_depth.squeeze(1) + eps_depth)
        
        # 3 物理不确定性惩罚
        # ==========================================
        # 1. 提取基础统计量
        # ==========================================
        mean_2d = self.net.err_2d_mean.to(self.device)
        std_2d = self.net.err_2d_std.to(self.device)
        
        mean_3d_raw = self.net.err_3d_mean.to(self.device)
        std_3d_raw = self.net.err_3d_std.to(self.device)
        std_depth_raw = self.net.depth_std.to(self.device) # 物理深度波动图

        # 适度膨胀基线，解决 1~2 像素的基础错位
        mean_3d_tolerant = F.max_pool2d(mean_3d_raw, kernel_size=5, stride=1, padding=2)
        std_3d_tolerant = F.max_pool2d(std_3d_raw, kernel_size=5, stride=1, padding=2)
        
        # ★ 引入强力全局锚点 Epsilon，彻底抹杀除零爆炸
        strong_eps_2d = std_2d.mean() * 0.5 + 1e-3
        strong_eps_3d = std_3d_tolerant.mean() * 0.5 + 1e-3
        
        # 计算基础 Z-Score
        err_2d_norm = F.relu(err_2d_smooth - mean_2d) / (std_2d + strong_eps_2d)
        err_3d_feat_norm = F.relu(err_3d_smooth - mean_3d_tolerant) / (std_3d_tolerant + strong_eps_3d)

        # ★ 补充被遗漏的 err_depth_norm 计算！
        eps_depth = torch.clamp(std_depth_raw.mean() * 0.1, min=1e-3)
        err_depth_raw = torch.abs(depth_for_eval - self.net.depth_mean.to(self.device))
        err_depth_norm = err_depth_raw.squeeze(1) / (std_depth_raw.squeeze(1) + eps_depth)

        # ==========================================
        # ★ 终极必杀技：物理不确定性惩罚 (Physical Uncertainty Penalty)
        # 不腐蚀掩码，直接利用数据集固有的深度波动来压制结构性飞点！
        # ==========================================
        # 将深度波动图膨胀一下，形成一个包裹把手和边缘的“不确定性光晕”
        std_depth_aura = F.max_pool2d(std_depth_raw, kernel_size=9, stride=1, padding=4)
        
        # 归一化到 0~1 之间，形成惩罚权重图
        depth_uncertainty = std_depth_aura / (std_depth_aura.max() + 1e-5)
        
        # 核心逻辑：
        # - 在平坦中心（病灶区）：uncertainty 接近 0，分母为 1，真实 3D 缺陷保留 100% 耀眼红光！
        # - 在左侧复杂把手：uncertainty 接近 1，分母为 6，该区域的 3D 误检得分被瞬间压制到 16%！
        err_3d_feat_norm = err_3d_feat_norm / (1.0 + 5.0 * depth_uncertainty.squeeze(1))
        
        # # 4实时断崖惩罚
        # # ==========================================
        # # 1. 提取基础统计量与防抖基线
        # # ==========================================
        # mean_2d = self.net.err_2d_mean.to(self.device)
        # std_2d = self.net.err_2d_std.to(self.device)
        
        # mean_3d_raw = self.net.err_3d_mean.to(self.device)
        # std_3d_raw = self.net.err_3d_std.to(self.device)

        # # 适度膨胀基线，解决 1~2 像素的基础空间错位
        # mean_3d_tolerant = F.max_pool2d(mean_3d_raw, kernel_size=5, stride=1, padding=2)
        # std_3d_tolerant = F.max_pool2d(std_3d_raw, kernel_size=5, stride=1, padding=2)
        
        # # ==========================================
        # # ★ 终极修复 1：提高防爆底线 (Crush Z-Score Explosion)
        # # 将 min 从 1e-3 提高到 0.05！彻底抹杀微小误差被除以极小 std 后放大的假阳性！
        # # ==========================================
        # strong_eps_2d = torch.clamp(std_2d.mean() * 0.5, min=0.01)
        # strong_eps_3d = torch.clamp(std_3d_tolerant.mean() * 0.5, min=0.05) 
        
        # # 计算基础 Z-Score
        # err_2d_norm = F.relu(err_2d_smooth - mean_2d) / (std_2d + strong_eps_2d)
        # err_3d_feat_norm = F.relu(err_3d_smooth - mean_3d_tolerant) / (std_3d_tolerant + strong_eps_3d)

        # # ==========================================
        # # ★ 终极修复 2：实时深度断崖惩罚 (Real-time Cliff Penalty)
        # # ==========================================
        # # 直接计算当前测试图深度的空间梯度 (Sobel-like)，找出当前物体的所有物理悬崖！
        # depth_pad = F.pad(depth_for_eval, (1, 1, 1, 1), mode='replicate')
        # dx = torch.abs(depth_pad[:, :, :, 2:] - depth_pad[:, :, :, :-2])
        # dy = torch.abs(depth_pad[:, :, 2:, :] - depth_pad[:, :, :-2, :])
        # depth_grad = dx[:, :, 1:-1, :] + dy[:, :, :, 1:-1]
        
        # # 膨胀梯度，形成包裹悬崖的“防误检光晕” (kernel=7 保护边缘免受 3D 撕裂影响)
        # cliff_halo = F.max_pool2d(depth_grad, kernel_size=7, stride=1, padding=3)
        # cliff_halo = cliff_halo / (cliff_halo.max() + 1e-5)
        
        # # 施加极其严厉的悬崖惩罚：在深度断崖（把手边缘）处，3D 得分被强行除以 10！
        # # 你的平坦中心病灶梯度几乎为 0，得分保留 100%；把手悬崖处梯度极大，红斑瞬间被压平！
        # err_3d_feat_norm = err_3d_feat_norm / (1.0 + 9.0 * cliff_halo.squeeze(1))

        # # ---------------- 补充物理深度异常 (Depth) 计算 ----------------
        # std_depth_raw = self.net.depth_std.to(self.device)
        # eps_depth = torch.clamp(std_depth_raw.mean() * 0.1, min=0.01)
        # err_depth_raw = torch.abs(depth_for_eval - self.net.depth_mean.to(self.device))
        # err_depth_norm = err_depth_raw.squeeze(1) / (std_depth_raw.squeeze(1) + eps_depth)
        

        # fg_mask_dilated = F.max_pool2d(fg_mask_raw, kernel_size=3, stride=1, padding=1)
        # fg_mask_eval = F.interpolate(fg_mask_dilated, size=(self.img_size_val, self.img_size_val), mode='nearest').squeeze(1)
        
        # # ==========================================
        # # ★ 终极防线：向内腐蚀掩码，彻底抛弃物理断崖边缘的 3D 畸变点！
        # # ==========================================
        # bg_mask_raw = 1.0 - fg_mask_raw
        # # 用 5x5 的核向内腐蚀掉 2 个像素的最危险边缘
        # eroded_bg_raw = F.max_pool2d(bg_mask_raw, kernel_size=9, stride=1, padding=4)
        # fg_mask_clean = 1.0 - eroded_bg_raw
        # fg_mask_eval = fg_mask_clean.squeeze(1) # 直接作为评估掩码
        
        # 发现不能向内扩散
        fg_mask_eval = fg_mask_raw.squeeze(1)
        
        
        err_2d_norm = err_2d_norm * fg_mask_eval
        err_3d_feat_norm = err_3d_feat_norm * fg_mask_eval
        err_depth_norm = err_depth_norm * fg_mask_eval

        fused = self.weight_2d * err_2d_norm + \
                self.weight_3d_feat * err_3d_feat_norm + \
                self.weight_depth * err_depth_norm

        bg_mask = 1.0 - fg_mask_raw
        eroded_bg = F.max_pool2d(bg_mask, kernel_size=5, stride=1, padding=2)
        score_mask = (1.0 - eroded_bg).squeeze(1).cpu()
        
        gt = gt.detach().cpu()
        label = label.detach().cpu()
        fused = fused.cpu()
        for b in range(B):
            gmap = gt[b, 0]           
            fmap = fused[b]          

            fmap_score = fmap * score_mask[b]
            fmap_flat = fmap_score.flatten()
            k_val = max(1, int(fmap_flat.numel() * 0.01)) 
            topk_vals, _ = torch.topk(fmap_flat, k_val)
            img_score = float(topk_vals.mean())
            self.image_preds.append(img_score)
            self.image_labels.append(int(label[b]))

            self.gts.append(gmap.numpy())
            self.pred_maps.append(fmap.clone())
            
            self.maps_2d.append(err_2d_norm[b].clone())
            self.maps_3d.append(err_3d_feat_norm[b].clone())

            if isinstance(sample, (tuple, list)):
                rgb_b   = sample[0][b:b+1].cpu()
                depth_b = sample[3][b:b+1].cpu() # 修正为3，因为 depth_map 是解包出来的第4个
                ps_b    = sample[4][b:b+1].cpu() # 修正为4，因为 ps 是第5个
            else:
                rgb_b = depth_b = ps_b = None

            self.vis_samples.append((rgb_b, depth_b, ps_b))

        return

    def calculate_metrics(self):
        from sklearn.metrics import roc_auc_score
        from utils.au_pro_util import calculate_au_pro

        if len(self.image_labels) == 0:
            self.image_rocauc = float("nan")
            self.pixel_rocauc = float("nan")
            self.au_pro = float("nan")
            return

        if len(set(self.image_labels)) > 1:
            self.image_rocauc = float(roc_auc_score(self.image_labels, self.image_preds))
        else:
            self.image_rocauc = float("nan")

        gts_flat = np.concatenate([gt.flatten() for gt in self.gts])
        preds_flat = np.concatenate([pred.cpu().numpy().flatten() for pred in self.pred_maps])
        
        # ==========================================
        # ★ 终极修复：强制二值化 Ground Truth (GT)
        # 将插值产生的平滑边缘一刀切，严格转换为 0 和 1，满足 sklearn 的二分类强迫症！
        # ==========================================
        # 如果你的 gt 是 0.0~1.0 的浮点数，就用 > 0.5；如果是 0~255，这里同样适用，都会变成 bool 然后转成 0/1
        gts_flat = (gts_flat > 0.5).astype(np.int8)
        
        assert gts_flat.shape == preds_flat.shape, f"Shape mismatch: GT {gts_flat.shape} vs Pred {preds_flat.shape}"
        self.pixel_rocauc = float(roc_auc_score(gts_flat, preds_flat))
