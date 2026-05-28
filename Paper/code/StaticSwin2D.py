class StaticSwin2D(nn.Module):
    """2D Swin Transformer"""
    def __init__(self, in_channels, out_dim, 
                 embed_dim=32, depths=[2, 2], num_heads=[2, 4],
                 window_size=5, norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 2D Swin Blocks（window_size时间维=1）
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=(1, window_size, window_size),
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
        x = self.patch_embed(x_stat)      # [B, embed_dim, H, W]
        x = x.unsqueeze(2)                    
        
        for layer in self.layers:
            x = layer(x.contiguous())     # [B, C, 1, H, W]
        
        x = x.squeeze(2)                  # [B, C, H, W]
        x = self.proj(x)                  # [B, out_dim, H, W]
        x = F.adaptive_avg_pool2d(x, target_size)
        return x
    
 