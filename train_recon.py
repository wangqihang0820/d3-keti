# train_recon.py
import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm  # 引入进度条
import random
import matplotlib.pyplot as plt  # ★ 新增：用于绘制 Loss 曲线



from dataset import (
    get_data_loader,
    test_3d_classes,
)
from feature_extractors.recon_features import ReconFeatures


# 设置随机种子，保证可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------
#  参数解析
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="2D-3D Reconstruction-based Anomaly Detection (FusionReconNet)"
    )

    # 数据相关
    parser.add_argument("--dataset_type",default="test_3d",type=str,choices=["mvtec3d", "eyecandies", "test_3d"],help="Dataset type.",)
    parser.add_argument("--dataset_path",type=str,required=True,help="Root path of dataset.",)
    parser.add_argument("--img_size_val",default=224,type=int,help="Input image size.",)

    # 编码器相关（保持和 main.py 一致）
    parser.add_argument("--rgb_backbone_name",default="vit_base_patch8_224_dino",type=str,choices=["vit_base_patch8_224_dino","vit_base_patch8_224","vit_base_patch8_224_in21k","vit_small_patch8_224_dino",],help="RGB backbone name (timm).",)
    parser.add_argument("--xyz_backbone_name",default="Point_MAE",type=str,choices=["Point_MAE", "Point_Bert", "FPFH"],help="3D backbone name.",)

    # DataLoader / 预处理配置（和 dataset.py 对齐）
    parser.add_argument("--downsampling",default=1,type=int,help="Downsampling factor for organized point cloud.",)
    parser.add_argument("--rotate_angle",default=0.0,type=float,help="Perspective rotation angle (if used).",)
    parser.add_argument("--small",action="store_true",help="Use a small subset of data for quick debug.",)

    # 模型结构的一些参数（ReconFeatures 里可能会用）
    parser.add_argument("--group_size",default=128,type=int,help="Group size for point grouping (if used in Model).",)
    parser.add_argument("--num_group",default=1024,type=int,help="Number of groups for point grouping.",)

    # 训练超参
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--batch_size", default=8, type=int)

    # Loss 权重
    parser.add_argument("--lambda_geo", default=0.1, type=float, help="Weight for geometric loss.")

    # 路径
    parser.add_argument("--save_dir", default="checkpoints_recon", type=str)
    parser.add_argument("--random_state", default=42, type=int)

    args = parser.parse_args()
    return args


def save_visualizations(save_root, idx, sample, mask, s_map, s_map_2d, s_map_3d):
    # ★ 2. 修正：确保目录存在
    os.makedirs(save_root, exist_ok=True)
    
    # rgb_tensor = sample[0]
    # ★ 消除 batch 维度，从 [1, 3, H, W] 变成 [3, H, W]
    rgb_tensor = sample[0].squeeze(0) if sample[0].dim() == 4 else sample[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_denorm = rgb_tensor.cpu() * std + mean
    rgb_np = (rgb_denorm.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    
    def render_heatmap(error_map):
        error_map = error_map.cpu().numpy() if isinstance(error_map, torch.Tensor) else error_map
        # if error_map.max() > 0:
        #     error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        # else:
        #     error_map = np.zeros_like(error_map)
        
        # ★ 终极修复：强行挤掉多余的通道维度，确保它是纯粹的 2D 数组 (H, W)
        error_map = np.squeeze(error_map)
        # ★ 核心修复：使用 99% 和 1% 分位数替代绝对的 max 和 min
        max_val = np.percentile(error_map, 99.5)
        min_val = np.percentile(error_map, 0.5)
        
        if max_val > min_val:
            # clip 限制在 0~1 之间
            error_map = np.clip((error_map - min_val) / (max_val - min_val), 0, 1)
        else:
            error_map = np.zeros_like(error_map)
        
        # 转换为 uint8 格式
        error_map_uint8 = (error_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(error_map_uint8, cv2.COLORMAP_JET)
        return cv2.resize(heatmap, (rgb_np.shape[1], rgb_np.shape[0]))

    heatmap_total = render_heatmap(s_map)
    heatmap_2d = render_heatmap(s_map_2d)
    heatmap_3d = render_heatmap(s_map_3d)
    
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    mask_vis = (mask_np * 255).astype(np.uint8)
    mask_vis = cv2.resize(mask_vis, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = cv2.addWeighted(rgb_bgr, 0.6, heatmap_total, 0.4, 0)
    
    cv2.imwrite(f"{save_root}/{idx:03d}_rgb.png", rgb_bgr)
    cv2.imwrite(f"{save_root}/{idx:03d}_gt.png", mask_vis)
    cv2.imwrite(f"{save_root}/{idx:03d}_err_total.png", heatmap_total)
    cv2.imwrite(f"{save_root}/{idx:03d}_err_2d.png", heatmap_2d)
    cv2.imwrite(f"{save_root}/{idx:03d}_err_3d.png", heatmap_3d)
    cv2.imwrite(f"{save_root}/{idx:03d}_overlay.png", overlay)

def train_recon(args, class_name):
    setup_seed(args.random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n[Training] Class: {class_name} | Batch: {args.batch_size}")

    model = ReconFeatures(args).to(device)
    model.train()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.net.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # ★ 新增：学习率调度器 (Cosine Annealing)
    # 让 LR 从 1e-4 逐渐降到 1e-6，有助于模型后期收敛更精细
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ★ 3. 修正：get_data_loader 必须能接收 batch_size
    # 请务必确保 dataset.py 已经按照我上面的建议修改了！
    train_loader = get_data_loader(
        split="train",
        class_name=class_name,
        img_size=args.img_size_val,
        args=args  # 这里 args 包含了 batch_size
    )
    
    # ==========================================
    # ★ 新增 1：初始化历史记录列表
    # ==========================================
    history = {
        'loss_total': [],
        'loss_2d': [],
        'loss_3d': [],
        'loss_geo': [],
        'ot_alpha': [],
        'ot_beta': []
    }

    for epoch in range(1, args.epochs + 1):
        # epoch_loss = 0.0
        # ★ 新增 2：初始化当前 Epoch 的累加器
        epoch_sums = {k: 0.0 for k in history.keys()}
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        
        for sample, _ in pbar:
            optimizer.zero_grad()
            rgb, xyz, depth, ps = sample
            rgb, xyz, ps = rgb.to(device), xyz.to(device), ps.to(device)
            
            # # Forward
            # rgb_recon, xyz_recon, rgb_target, F_gt, loss_geo, U, mask, center = model.net(rgb, xyz, ps)
            
            # # Loss
            # loss_2d = model.compute_hybrid_loss(rgb_recon, rgb_target.detach(), dim=1)
            # loss_3d = model.compute_hybrid_loss(xyz_recon, F_gt.detach(), dim=2)
            # loss_total = loss_2d + loss_3d + args.lambda_geo * loss_geo
            
            # loss_geo_raw就是还没乘以系数的，用作调试
            # loss_total,loss_2d,loss_3d,loss_geo,loss_geo_raw = model.train_step(sample)
            # 调用 train_step
            loss_dict = model.train_step(sample)
            
            loss_total = loss_dict['loss']
            loss_total.backward()
            # ★ 新增：梯度裁剪 (Gradient Clipping)
            # 强制将网络所有参数的梯度最大范数限制在 1.0，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # epoch_loss += loss_total.item()
            # ★ 新增 3：累加当前 batch 的指标
            epoch_sums['loss_total'] += loss_total.item()
            epoch_sums['loss_2d'] += loss_dict['l2d']
            epoch_sums['loss_3d'] += loss_dict['l3d']
            epoch_sums['loss_geo'] += loss_dict['geo']
            epoch_sums['ot_alpha'] += loss_dict['alpha']
            epoch_sums['ot_beta'] += loss_dict['beta']
            steps += 1
            
            # 显示 OT 动态调整的权重
            # 你会看到 alpha 和 beta 随着 epoch 变化，这证明 OT 在工作
            pbar.set_postfix({
                "Loss": f"{loss_total.item():.4f}",
                "2D": f"{loss_dict['l2d']:.3f}",
                "3D": f"{loss_dict['l3d']:.3f}",
                "geo": f"{loss_dict['geo']:.3f}",
                "α": f"{loss_dict['alpha']:.2f}", 
                "β": f"{loss_dict['beta']:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 更新学习率
        scheduler.step()
        
        # ★ 新增 4：计算该 Epoch 的平均值并存入历史记录
        for k in history.keys():
            history[k].append(epoch_sums[k] / steps)

    # === 150 轮结束 ===
    # ★ 在保存模型权重之前，强制遍历一次训练集，提取像素级基线记忆
    print("\n[Final Stage] Initializing Z-Score Database...")
    model.build_error_statistics(train_loader)

    ckpt_path = os.path.join(args.save_dir, f"{class_name}_recon.pth")
    torch.save(model.net.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")
    
    # ★ 新增 5：训练结束后，自动绘制并保存曲线图
    plot_training_curves(history, args.save_dir, class_name)

# def train_recon(args, class_name):
#     setup_seed(args.random_state)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs(args.save_dir, exist_ok=True)
    
#     print(f"\n[Training] Class: {class_name} | Batch: {args.batch_size}")

#     model = ReconFeatures(args).to(device)
#     model.train()
    
#     optimizer = torch.optim.AdamW(
#         model.net.parameters(), 
#         lr=args.lr, 
#         weight_decay=args.weight_decay
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

#     train_loader = get_data_loader(
#         split="train",
#         class_name=class_name,
#         img_size=args.img_size_val,
#         args=args
#     )

#     # ========================================================
#     # ★ 单 Batch 过拟合极速测试法 (开始)
#     # ========================================================
#     print("\n🚀 开始执行单 Batch 过拟合测试 (预计耗时: 2分钟)...")
    
#     # 1. 强行从 loader 中抽出第一个 batch (只取一次)
#     single_batch_data = next(iter(train_loader))
    
#     # 2. 我们只跑 50 轮
#     for epoch in range(1, 51):
#         optimizer.zero_grad()
        
#         # 3. 每轮都用这死死固定的同一个 batch
#         sample, _ = single_batch_data 
        
#         # 注意：这里直接调用你原来的 train_step，不需要自己写 forward
#         loss_dict = model.train_step(sample)
        
#         loss_total = loss_dict['loss']
#         loss_total.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         # 4. 直接打印日志看变化
#         print(f"Epoch [{epoch:02d}/50] "
#               f"Total Loss: {loss_total.item():.4f} | "
#               f"2D Loss: {loss_dict['l2d']:.4f} | "
#               f"3D Loss: {loss_dict['l3d']:.4f} | "
#               f"α: {loss_dict['alpha']:.2f} | β: {loss_dict['beta']:.2f}")
        
#         scheduler.step()

#     print("\n✅ 单 Batch 测试结束！请观察上面的 Loss 变化。")
#     import sys
#     sys.exit(0) # 强制退出，不保存模型，不跑评估
#     # ========================================================
#     # ★ 单 Batch 过拟合极速测试法 (结束)
#     # ========================================================

def plot_training_curves(history, save_dir, class_name):
    """
    绘制并保存训练过程中的各项 Loss 和 OT 权重走向
    """
    epochs = range(1, len(history['loss_total']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制总 Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss_total'], label='Total Loss', color='black', linewidth=2)
    plt.title(f'[{class_name}] Total Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 2. 绘制 2D 和 3D 的子 Loss (观察谁更难收敛)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['loss_2d'], label='2D Loss (RGB)', color='red')
    plt.plot(epochs, history['loss_3d'], label='3D Loss (XYZ)', color='blue')
    plt.title('2D vs 3D Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 3. 绘制 几何正则化 Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['loss_geo'], label='Geo Loss (SharedBasis)', color='green')
    plt.title('Geometric Regularization Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 4. 绘制 OT 动态权重分配 (观察模型侧重方向)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['ot_alpha'], label='Alpha (2D Weight)', color='orange')
    plt.plot(epochs, history['ot_beta'], label='Beta (3D Weight)', color='purple')
    plt.title('Optimal Transport (OT) Weights Dynamics')
    plt.xlabel('Epochs')
    plt.ylabel('Weight')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f"{class_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {plot_path}")

@torch.no_grad()
def eval_recon(args, class_name):
    # ... (这部分逻辑和之前一样，无需大改)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(args.save_dir, f"{class_name}_recon.pth")
    
    if not os.path.exists(ckpt_path):
        return

    print(f"\n[Evaluating] Class: {class_name}")
    model = ReconFeatures(args).to(device)
    model.net.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.reset_buffers()

    test_loader = get_data_loader(
        split="test",
        class_name=class_name,
        img_size=args.img_size_val,
        args=args
    )
    
    for sample, gt, label, rgb_path in tqdm(test_loader, desc="Inference"):
        model.predict(sample, gt, label, rgb_path)
    
    model.calculate_metrics()
    print(f"[{class_name}] Image AUC: {model.image_rocauc:.4f} | Pixel AUC: {model.pixel_rocauc:.4f}")
    
    vis_dir = os.path.join(args.save_dir, "vis", class_name)
    print(f"Saving visualization to {vis_dir} ...")
    # ★ 目录创建已移至 save_visualizations 内部，这里也可以留着
    os.makedirs(vis_dir, exist_ok=True)
    
    # ... (保存前20张的循环)
    for idx in tqdm(range(len(model.vis_samples)), desc="Saving Images"):
        sample = model.vis_samples[idx]
        mask = model.gts[idx]
        s_map = model.pred_maps[idx]
        # 兼容性处理
        s_map_2d = model.maps_2d[idx] if hasattr(model, 'maps_2d') else torch.zeros_like(s_map)
        s_map_3d = model.maps_3d[idx] if hasattr(model, 'maps_3d') else torch.zeros_like(s_map)
        
        save_visualizations(vis_dir, idx, sample, mask, s_map, s_map_2d, s_map_3d)

if __name__ == "__main__":
    args = parse_args()
    if args.dataset_type == "mvtec3d":
        classes = mvtec3d_classes()
    elif args.dataset_type == "test_3d":
        classes = test_3d_classes()
    else:
        classes = [args.dataset_type]

    for cls in classes:
        train_recon(args, class_name=cls)
        eval_recon(args, class_name=cls)
        
        
# python train_recon.py \
#     --dataset_path /media/li/data21/wqh/d3-keti/datasets_noval \
#     --dataset_type test_3d \
#     --batch_size 16 \
#     --epochs 200 \
#     --lr 0.0001 \
#     --save_dir ./checkpoints_v1
