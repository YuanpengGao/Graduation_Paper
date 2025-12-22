import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import gc # 垃圾回收

class FireDataset(Dataset):
    def __init__(self, nc_file, target_col="burned_areas"):
        super().__init__()
        
        print(f"🔄 正在加载数据 (内存优化模式): {nc_file} ...")
        
        # 1. 打开文件
        ds = xr.open_dataset(nc_file)
        self.feature_names = ds['variable'].values.tolist()
        self.target_idx = self.feature_names.index(target_col)
        
        # 2. 读取数据 (此时占用约 8.6 GB 内存)
        # 注意：这一步是内存占用的高峰
        self.data = ds['data'].values.astype(np.float32)
        ds.close() # 立即关闭文件句柄
        
        print(f"📊 原始数据加载完成，形状: {self.data.shape}")
        print("🧪 正在进行【原地】归一化 (节省内存)...")

        # -------------------------------------------------------
        # 🚀 内存优化的核心：原地操作 (In-place)
        # 不要写 data = (data - mean) / std，那会产生新副本！
        # -------------------------------------------------------

        # 3. 计算统计量 (使用 float32 节省计算内存)
        # 仅计算非 Target 通道的均值/方差
        # 这里的切片只是视图，不占内存
        features_view = self.data # 此时包含 target
        
        mean = np.nanmean(features_view, axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
        std = np.nanstd(features_view, axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
        std[std == 0] = 1.0
        
        # 备份 Target (因为归一化会破坏 0/1 标签)
        # 这会占用约 1GB 内存 (8000x10x128x128x4bytes)
        print("   - 备份 Target...")
        target_backup = self.data[:, :, self.target_idx, :, :].copy()
        
        # 4. 原地减均值 (In-place Subtraction)
        # 内存不会增加
        print("   - 执行减法...")
        np.subtract(self.data, mean, out=self.data)
        
        # 5. 原地除标准差 (In-place Division)
        # 内存不会增加
        print("   - 执行除法...")
        np.divide(self.data, std, out=self.data)
        
        # 6. 处理 NaN (原地替换)
        if np.isnan(self.data).any():
            print("   - 清洗 NaN...")
            np.nan_to_num(self.data, copy=False, nan=0.0)

        # 7. 还原 Target
        print("   - 还原 Target...")
        self.data[:, :, self.target_idx, :, :] = target_backup
        
        # 释放备份，回收 1GB 内存
        del target_backup
        gc.collect()
        
        print("✅ 数据加载与归一化完成，内存状态健康。")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 此时数据已经在内存里了，读取速度是纳秒级的
        # 直接拿来用
        sample = self.data[idx]
        # 🔥【修复数据泄露】
        # Input (x): 取前 9 天 (索引 0 到 -2)
        # Target (y): 取最后 1 天 (索引 -1)
        
        # x: (9, 8, 128, 128)
        x_data = sample[:-1] 
        
        # y: (1, 128, 128) -> 只取 Target 通道
        y_data = sample[-1, self.target_idx, :, :]
        
        x = torch.from_numpy(x_data).float()
        y = torch.from_numpy(y_data).float().unsqueeze(0)

        return x, y