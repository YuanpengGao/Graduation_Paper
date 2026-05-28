class GeoCrossAttention(nn.Module):
    """ 地理惩罚交叉注意力机制 (加入物理风场拉伸) """
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
        # omega 自动学习风场偏置强度
        self.omega = nn.Parameter(torch.tensor(0.1))

    def forward(self, q_x, kv_x, wind_u=None, wind_v=None):
        B, C, H, W = q_x.shape
        N = H * W

        # 动态构建欧氏距离矩阵
        coords_h = torch.arange(H, device=q_x.device).float()
        coords_w = torch.arange(W, device=q_x.device).float()
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1) # [2, N]
        
        # [2, N_query, N_key] 相对空间向量
        diff = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        dy = diff[0] # [N, N]
        dx = diff[1] # [N, N]
        
        dist_matrix = torch.sqrt(dx**2 + dy**2)
        penalty = F.softplus(self.tau) * dist_matrix.unsqueeze(0).unsqueeze(0)

        # 展平进行 Attention
        q_flat = q_x.view(B, C, N).transpose(1, 2)   
        kv_flat = kv_x.view(B, C, N).transpose(1, 2) 

        q = self.q_proj(q_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_flat).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if wind_u is not None and wind_v is not None:
            u_flat = wind_u.view(B, N, 1) 
            v_flat = wind_v.view(B, N, 1) 
            
            # 计算风场向量 (U, -V) 与 相对坐标向量 (dx, dy) 的点积
            wind_bias = u_flat * dx.unsqueeze(0) + v_flat * (-dy.unsqueeze(0))
            
            wind_bias = wind_bias.unsqueeze(1) # 变成 [B, 1, N, N]
            # 将风偏置叠加到 Attention 矩阵上
            attn = attn + F.softplus(self.omega) * wind_bias
            
        # 减去距离惩罚
        attn = attn - penalty
        attn_weights = self.dropout(attn.softmax(dim=-1))
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = q_flat + self.out_proj(out)
        return out.transpose(1, 2).view(B, C, H, W)

