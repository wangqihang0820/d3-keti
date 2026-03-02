import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

# Model 类是一个组合模型，它结合了 RGB Backbone 和 XYZ Backbone 两种特征提取方法，分别用于处理RGB图像和3D点云数据
class Model(torch.nn.Module):

    # 使用 timm 库创建一个预训练的RGB图像处理模型，默认使用的是 vit_base_patch8_224_dino 模型（基于 Vision Transformer）
    # self.xyz_backbone：根据输入的模型名称选择不同的3D点云特征提取方法（如 Point_MAE、Point-BERT 或 FPFH）。这些方法分别用于处理点云数据。
    # checkpoint_path='/fuxi_team14_intern/m3dm/checkpoints/dino_vitbase8_pretrain.pth',
    def __init__(self, device, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None,
                 checkpoint_path='checkpoints/dino_vitbase8_pretrain.pth',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024):
        super().__init__()
        # 'vit_base_patch8_224_dino'
        # Determine if to output features.
        self.device = device

        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})

        ## RGB backbone
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=False,
                                              checkpoint_path=checkpoint_path,
                                              **kwargs)
        # self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True,
        #                                       checkpoint_path=checkpoint_path,
        #                                       **{k: v for k, v in kwargs.items() if k != 'checkpoint_path'})

        ## XYZ backbone

        if xyz_backbone_name == 'Point_MAE':
            self.xyz_backbone = PointTransformer(group_size=group_size, num_group=num_group)
            # self.xyz_backbone.load_model_from_ckpt("/fuxi_team14_intern/m3dm/checkpoints/pointmae_pretrain.pth")
            self.xyz_backbone.load_model_from_ckpt("checkpoints/pointmae_pretrain.pth")
        elif xyz_backbone_name == 'Point-BERT':
            self.xyz_backbone=PointTransformer(group_size=group_size, num_group=num_group, encoder_dims=256)
            self.xyz_backbone.load_model_from_pb_ckpt("/fuxi_team14_intern/m3dm/checkpoints/Point-BERT.pth")
        elif xyz_backbone_name == 'FPFH':
            self.xyz_backbone=FPFH(group_size=group_size, num_group=num_group,voxel_size=0.05)
            #self.xyz_backbone.load_model_from_pb_ckpt("/workspace/data2/checkpoints/Point-BERT.pth")



# 提取输入的RGB图像的特征。它通过 patch_embed 和 norm_pre 进行嵌入和预处理，然后通过多个Transformer块提取特征，并最终将特征维度转换为28x28的形状。
    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        if self.rgb_backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.rgb_backbone.blocks(x)
        x = self.rgb_backbone.norm(x)

        B = x.shape[0]
        feat = x[:, 1:].permute(0, 2, 1).reshape(B, -1, 28, 28)
        return feat


# 将输入的RGB和XYZ数据通过相应的backbone模型处理后，返回RGB特征、XYZ特征和其他相关信息。
    def forward(self, rgb, xyz):
        
        rgb_features = self.forward_rgb_features(rgb)

        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz)

        xyz_features.permute(0, 2, 1)

        return rgb_features, xyz_features, center, ori_idx, center_idx


# 从输入点云数据中选择出具有代表性的点。使用了最远点采样（FPS）算法，该算法帮助选取分布均匀的样本点，从而有效地减少数据量并保持点云的几何特征。
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx

# Group 类用于将输入的点云分为多个小组，每个小组包含一定数量的点。它通过 最远点采样（FPS）选择小组中心点，然后使用 KNN 算法找到与这些中心点最邻近的点，形成点云的局部区域。
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = fps(xyz.contiguous(), self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


# 定义了一个简单的卷积神经网络，用于提取局部点云的特征。网络包括两个卷积层，分别用于从输入的点云中提取低级和高级特征。
class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    # 接收分组的点云数据，通过两个卷积层提取特征，并返回全局特征
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 实现了标准的 多头自注意力机制。该类包括计算查询（Q）、键（K）和值（V）的线性变换，以及计算注意力权重并应用于值的过程。
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer中的基本构建块，它结合了 自注意力机制 和MLP层
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    # 通过自注意力和MLP层分别对输入进行处理，并加上残差连接
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# 由多个 Block 组成，是Transformer编码器的核心部分
class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    # 将输入通过多个 Block 进行处理，并返回每一层的特征
    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


# 用于处理点云数据并通过Transformer对其进行编码
class PointTransformer(nn.Module):
    def __init__(self, group_size=128, num_group=1024, encoder_dims=384):
        super().__init__()

        # trans_dim: Transformer 的嵌入维度，默认为 384。
        self.trans_dim = 384
        # Transformer 编码器的深度，默认为 12，表示 Transformer 中堆叠的 Block 数量。
        self.depth = 12
        self.drop_path_rate = 0.1
        # Transformer 的多头注意力机制中的头数，默认为 6
        self.num_heads = 6

        # 用于分割点云数据的超参数，group_size 表示每个小组的点数，num_group 表示组数。
        self.group_size = group_size
        self.num_group = num_group
        # grouper 该模型首先通过 Group 模块将输入的点云数据分割成多个小组。
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        # 点云特征编码器的维度，默认为 384。
        self.encoder_dims = encoder_dims
        if self.encoder_dims != self.trans_dim:
            # 用于 Transformer 的分类标记（CLS token）和位置编码（position encoding）。
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        # 编码器模块，用于提取点云数据的特征
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        # pos_embed: 位置嵌入，将点云中的每个点的位置映射到高维空间，以便 Transformer 处理。
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    # 从预训练的 MAE 模型中加载权重，适用于加载基于自编码器的预训练模型。
    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            #if incompatible.missing_keys:
            #    print('missing_keys')
            #    print(
            #            incompatible.missing_keys
            #        )
            #if incompatible.unexpected_keys:
            #    print('unexpected_keys')
            #    print(
            #            incompatible.unexpected_keys

            #        )

            # print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    # 从 Point-BERT 模型中加载权重
    def load_model_from_pb_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                    incompatible.missing_keys
                )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                    incompatible.unexpected_keys

                )
                
        print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        # """
        # 入口统一兼容不同 xyz 形状，然后按原来的 Point_MAE 流程跑：
        # - 分组 (Group)
        # - 局部 Encoder
        # - Transformer 编码
        # - 拼接多层特征，输出: (B, 1152, G), center, ori_idx, center_idx
        # """
        # 1) 先统一成 (B, N, 3)
        pts = self._to_BN3(pts)          # (B, N, 3)

        # 2) FPS + KNN 分组
        neighborhood, center, ori_idx, center_idx = self.group_divider(pts)  # neighborhood: (B, G, M, 3)

        # 3) 编码 + Transformer（基本沿用你原来的逻辑，只是去掉了对 (B, C, N) 的硬编码）
        if self.encoder_dims != self.trans_dim:
            # Encoder: B G N 3 -> B G Cenc
            group_input_tokens = self.encoder(neighborhood)  # (B, G, encoder_dims)
            group_input_tokens = self.reduce_dim(group_input_tokens)  # (B, G, trans_dim)

            # CLS token & pos embed
            cls_tokens = self.cls_token.expand(group_input_tokens.size(0), 1, -1)  # (B, 1, trans_dim)
            cls_pos = self.cls_pos.expand(group_input_tokens.size(0), 1, -1)

            pos = self.pos_embed(center)  # (B, G, trans_dim)

            x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # (B, 1+G, trans_dim)
            pos = torch.cat((cls_pos, pos), dim=1)                  # (B, 1+G, trans_dim)

            feature_list = self.blocks(x, pos)  # list of (B, 1+G, trans_dim)
            # 去掉 CLS，只保留 group token，转成 (B, C, G)
            feature_list = [
                self.norm(feat)[:, 1:].transpose(-1, -2).contiguous()
                for feat in feature_list
            ]  # 每个: (B, trans_dim, G)

            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # (B, 1152, G)
            return x, center, ori_idx, center_idx
        else:
            # encoder_dims == trans_dim 的分支
            group_input_tokens = self.encoder(neighborhood)  # (B, G, trans_dim)
            pos = self.pos_embed(center)                    # (B, G, trans_dim)
            x = group_input_tokens                           # (B, G, trans_dim)

            feature_list = self.blocks(x, pos)               # list of (B, G, trans_dim)
            feature_list = [
                self.norm(feat).transpose(-1, -2).contiguous()
                for feat in feature_list
            ]  # 每个: (B, trans_dim, G)

            x = torch.cat((feature_list[0], feature_list[1], feature_list[2]), dim=1)  # (B, 1152, G)
            return x, center, ori_idx, center_idx

# -------- 新增：统一把输入整理成 (B, N, 3) --------
    def _to_BN3(self, pts: torch.Tensor) -> torch.Tensor:
        # 将各种常见 xyz 形状转成 (B, N, 3)，方便 group_divider 使用。
        # 支持：
        # - (B, 3, N)
        # - (B, N, 3)
        # - (B, 3, H, W)
        # - (B, H, W, 3)
        if pts.dim() == 3:
            # (B, 3, N) or (B, N, 3)
            if pts.size(1) == 3 and pts.size(2) != 3:
                # (B, 3, N) -> (B, N, 3)
                pts_bn3 = pts.transpose(1, 2).contiguous()
            elif pts.size(2) == 3:
                # (B, N, 3) 直接用
                pts_bn3 = pts
            else:
                raise RuntimeError(f"Unexpected 3D xyz shape: {pts.shape}")
        elif pts.dim() == 4:
            # (B, 3, H, W) or (B, H, W, 3)
            if pts.size(1) == 3:
                # (B, 3, H, W) -> (B, 3, H*W) -> (B, H*W, 3)
                B, C, H, W = pts.shape
                pts_bn3 = pts.view(B, C, -1).transpose(1, 2).contiguous()
            elif pts.size(-1) == 3:
                # (B, H, W, 3) -> (B, H*W, 3)
                B, H, W, C = pts.shape
                assert C == 3, f"Unexpected xyz last dim != 3: {pts.shape}"
                pts_bn3 = pts.view(B, H * W, C).contiguous()
            else:
                raise RuntimeError(f"Unexpected 4D xyz shape: {pts.shape}")
        else:
            raise RuntimeError(f"Unsupported xyz dim: {pts.dim()}, shape={pts.shape}")

        return pts_bn3

