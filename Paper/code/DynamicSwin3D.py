class DynamicSwin3D(nn.Module):
    """ 3D Swin，负责动态变量提取 """
    def __init__(self, patch_size=(1, 1, 1), in_chans=10, embed_dim=32,
                  depths=[2, 2, 8], num_heads=[2, 4, 8], window_size=(2, 5, 5),
                  norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans,
                                         embed_dim=embed_dim, norm_layer=norm_layer)
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

