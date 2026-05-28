import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul

np.set_printoptions(suppress=True, precision=2)

# =====================================================================
# 1. 底层基础模块 (保持原版 SwinTransformer3D 的精简结构)
# =====================================================================

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
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    use_shift_size = list(shift_size) if shift_size is not None else None
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0
    return tuple(use_window_size) if use_shift_size is None else (tuple(use_window_size), tuple(use_shift_size))

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 修复 meshgrid 警告
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        shortcut = x
        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, dim * 2, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x):
        B, D, H, W, C = x.shape
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        if (D % 2 == 1) and D != 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, D % 2))
            
        if D == 1:
            x0 = x[:, :, 0::2, 0::2, :]
            x1 = x[:, :, 1::2, 0::2, :]
            x2 = x[:, :, 0::2, 1::2, :]
            x3 = x[:, :, 1::2, 1::2, :]
        else:
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 1::2, 0::2, :]
            x2 = x[:, 0::2, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return self.reduction(self.norm(x))

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(2, 7, 7), mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(dim=dim//2, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        # 严格的安全通道重排：[B, C, D, H, W] -> [B, D, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        if self.downsample is not None:
            x = self.downsample(x)

        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)
            
        # 恢复 [B, C, D, H, W]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0: x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0: x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0: x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            Dd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x).transpose(1, 2).view(-1, self.embed_dim, Dd, Wh, Ww)
        return x


# =====================================================================
# 2. 新增与解耦模块：静态 CNN、动态 Swin、以及 Geo-CA
# =====================================================================

class StaticSwin2D(nn.Module):
    """用 2D Swin Transformer 替换原来的 CNN 静态编码器"""
    def __init__(self, in_channels, out_dim, 
                 embed_dim=32, depths=[2, 2], num_heads=[2, 4],
                 window_size=5, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Patch Embedding：用 2D Conv 把 [B, C, H, W] -> [B, embed_dim, H, W]
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 2D Swin Blocks（借用已有的3D结构降维到2D，window_size时间维=1）
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=(1, window_size, window_size),  # 时间维=1，退化成2D
                drop=0.3, attn_drop=0., drop_path=0.1,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer > 0 else None
            )
            self.layers.append(layer)
        
        num_features = int(embed_dim * 2 ** (len(depths) - 1))
        
        # 投影到 out_dim
        self.proj = nn.Conv2d(num_features, out_dim, kernel_size=1)
        
    def forward(self, x_stat, target_size):
        # x_stat: [B, C_stat, H, W]
        x = self.patch_embed(x_stat)           # [B, embed_dim, H, W]
        x = x.unsqueeze(2)                     # [B, embed_dim, 1, H, W] 伪3D
        
        for layer in self.layers:
            x = layer(x.contiguous())          # [B, C, 1, H, W]
        
        x = x.squeeze(2)                       # [B, C, H, W]
        x = self.proj(x)                       # [B, out_dim, H, W]
        x = F.adaptive_avg_pool2d(x, target_size)
        return x
    
class DynamicSwin3D(nn.Module):
    """ 纯净版 3D Swin，只负责动态气象提取，剥离了任何提前拼接的逻辑 """
    def __init__(self, patch_size=(1, 1, 1), in_chans=10, embed_dim=32, depths=[2, 2, 8], num_heads=[2, 4, 8], window_size=(2, 5, 5), norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=0.3)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                drop=0.3, attn_drop=0., drop_path=0.3,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer > 0 else None
            )
            self.layers.append(layer)
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_d):
        x = self.patch_embed(x_d)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x.contiguous())
        return x  # 返回 [B, C, D_out, H_out, W_out]

class GeoCrossAttention(nn.Module):
    """ 地理惩罚交叉注意力机制 """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)

        # tau 自动学习地理衰减程度
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, q_x, kv_x):
        # q_x (动态), kv_x (静态): [B, C, H, W]
        B, C, H, W = q_x.shape
        N = H * W

        # 动态构建欧氏距离矩阵
        coords_h = torch.arange(H, device=q_x.device).float()
        coords_w = torch.arange(W, device=q_x.device).float()
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        diff = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        dist_matrix = torch.sqrt((diff ** 2).sum(dim=0))  # [N, N]

        # 展平进行 Attention
        q_flat = q_x.view(B, C, N).transpose(1, 2)   # [B, N, C]
        kv_flat = kv_x.view(B, C, N).transpose(1, 2) # [B, N, C]

        q = self.q_proj(q_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 施加地理惩罚：距离越远，减去的 penalty 越大，注意力趋近于 0
        penalty = F.softplus(self.tau) * dist_matrix.unsqueeze(0).unsqueeze(0)
        attn = attn - penalty

        attn_weights = self.dropout(attn.softmax(dim=-1))
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        
        # 残差连接
        out = q_flat + self.out_proj(out)

        return out.transpose(1, 2).view(B, C, H, W)


# =====================================================================
# 3. 最终顶层 Wrapper：组合双分支与融合模块
# =====================================================================

class DualBranchSwin2d3d(nn.Module):
    """ 
    采用 3x3 核心 ROI 感知区的双分支架构 (完美平衡 Precision 与 Recall)
    """
    def __init__(self, input_channels_d=10, input_channels_s=5, input_channels_c=10, n_classes=2, **kwargs):
        super().__init__()
        
        dynamic_chans = input_channels_d
        static_chans = input_channels_s + input_channels_c 
        
        # 1. 动态气象提取器 (3D)
        self.branch_dynamic = DynamicSwin3D(in_chans=dynamic_chans)
        self.embed_dim = self.branch_dynamic.num_features 
        
        # 2. 静态地形提取器 (2D)
        self.branch_static = StaticSwin2D(in_channels=static_chans, out_dim=self.embed_dim)
        
        # 3. 地理感知交叉注意力
        self.geo_ca = GeoCrossAttention(dim=self.embed_dim)

        # 4. 核心分类头 (输入维度固定为 embed_dim * 3 * 3)
        # 相比全图 Flatten 的 6000 多维，这里只有 128 * 9 = 1152 维，既有空间感又不易过拟合
        # SwinGCA_0 无
        roi_flatten_dim = self.embed_dim * 9 
        
        self.head = nn.Sequential(
            # SwinGCA_0 全局平均
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # SwinGCA 2
            # nn.Linear(roi_flatten_dim, 256),
            # nn.LayerNorm(256), 
            # nn.GELU(),
            # nn.Dropout(0.4),  
            # nn.Linear(256, 64),
            # nn.GELU(),
            
            # nn.Dropout(0.2),
            nn.Linear(64, n_classes),
            nn.LogSoftmax(dim=1) 
        )

    def forward(self, data_s, data_c, data_d, data_t=None):

        # 使用 torch.flip 将其从 [T, T-1, ..., T-9] 翻转为 [T-9, T-8, ..., T]
        data_d = torch.flip(data_d, dims=[2])
        
        x_stat = torch.cat((data_s, data_c), dim=1)
        x_dyn_3d = self.branch_dynamic(data_d)  
        x_dyn_2d = x_dyn_3d.mean(dim=2)         
        
        B, C, H_out, W_out = x_dyn_2d.shape
        x_stat_2d = self.branch_static(x_stat, target_size=(H_out, W_out)) 
        
        # [B, C, H_out, W_out]
        x_fused = self.geo_ca(q_x=x_dyn_2d, kv_x=x_stat_2d)
        
        '''
        # SwinGCA_2 
        # ================= 核心绝杀：3x3 核心威胁区截取 =================
        center_h, center_w = H_out // 2, W_out // 2
        
        # 切片提取中心 3x3 区域 (涵盖周围 8 个方向的逼近火情)
        # 假设 H_out, W_out >= 3，这对 Swin 下采样后(通常是7x7)是绝对满足的
        x_roi = x_fused[:, :, center_h-1 : center_h+2, center_w-1 : center_w+2] # 变成 [B, C, 3, 3]
        
        # 展平这个核心区
        x_roi_flat = x_roi.reshape(B, -1) # 变成 [B, C * 9]
        # ==============================================================
        '''
        # 步骤 6：输出火灾概率
        # out = self.head(x_roi_flat)

        # SwinGCA_0 直接输出x_fused
        out = self.head(x_fused)
        
        return out
# =====================================================================
# 测试脚本
# =====================================================================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 模拟输入：假设 dynamic_chans 为 10
    test_d = torch.randn((4, 10, 10, 25, 25)).to(device)
    test_s = torch.randn((4, 5, 25, 25)).to(device)
    test_c = torch.randn((4, 10, 25, 25)).to(device)

    # 实例化双分支 GCA 模型
    model = DualBranchSwin2d3d(static_chans=15, dynamic_chans=10, num_classes=2).to(device)

    # 前向传播测试
    out = model(test_s, test_c, test_d)
    
    print("\n[✓] 前向传播成功!")
    print(f"预测输出形状: {out.shape} (期望 [Batch, 2])")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")