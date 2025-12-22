import xarray as xr
import numpy as np
import tqdm
import os
import pandas as pd

# ================= 配置区 =================
BIG_FILE_PATH = "/root/autodl-tmp/Graduation_Paper/data/dataset_greece.nc"
INDEX_FILE_PATH = "./fire_indices.npy"
OUTPUT_NC_PATH = "./data/balanced_train_data_2000.nc"

SEQ_LEN = 10      # 过去10天
PATCH_SIZE = 128
HALF_SIZE = PATCH_SIZE // 2

FEATURE_NAMES = [
    'burned_areas', 'ndvi', 'fwi', 
    'max_t2m', 'max_wind_speed', 
    'CLC_2018_6', 'CLC_2018_7', 'CLC_2018_8'
]
# =========================================

def extract_surgical():
    if not os.path.exists(INDEX_FILE_PATH):
        print("❌ 找不到索引文件！")
        return

    print("📖 正在加载索引文件...")
    indices = np.load(INDEX_FILE_PATH)
    
    # 1. 筛选逻辑
    df = pd.DataFrame(indices, columns=['t', 'y', 'x'])
    df = df.sort_values('t') # 稍微排个序，让硬盘磁头少跳一点

    # ✂️ 只取 500 个用于快速验证
    if len(df) > 2000:
        print(f"✂️ 正在随机抽取 5000 个样本...")
        df = df.sample(n=2000, random_state=42).sort_values('t')
    
    total_samples = len(df)
    print(f"🔥 准备提取 {total_samples} 个样本 (外科手术模式)...")

    # 2. 预分配数组
    num_vars = len(FEATURE_NAMES)
    all_data = np.zeros((total_samples, SEQ_LEN, num_vars, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    # 3. 打开大文件 (不加载数据，只建立连接)
    # chunks=None 或 {} 表示使用文件原生的分块方式
    ds = xr.open_dataset(BIG_FILE_PATH, chunks={})
    
    # 4. 逐个提取 (Surgical Extraction)
    # 因为只需要读极小的数据量 (128x128)，所以循环 500 次会非常快
    
    print("🚀 开始极速切割...")
    success_count = 0
    
    # 使用 tqdm 显示进度
    for i, (_, row) in tqdm.tqdm(enumerate(df.iterrows()), total=total_samples):
        t_center = int(row['t'])
        y_center = int(row['y'])
        x_center = int(row['x'])
        
        # 计算切片范围
        t_start = t_center - SEQ_LEN + 1
        t_end = t_center + 1 # xarray slice是包含结尾的，但isel如果不含需要注意
        
        y_min = y_center - HALF_SIZE
        y_max = y_center + HALF_SIZE
        x_min = x_center - HALF_SIZE
        x_max = x_center + HALF_SIZE
        
        try:
            # --- 核心：只读取这一小块 ---
            # 我们直接在硬盘上定位到这个小方块，只读这几MB数据
            patch_xr = ds[FEATURE_NAMES].isel(
                time=slice(t_start, t_end),
                y=slice(y_min, y_max),
                x=slice(x_min, x_max)
            )
            
            # 转 Numpy (这一步才会触发硬盘 I/O)
            patch_val = patch_xr.to_array(dim='variable').values
            # 形状: (Vars, Time, H, W) -> 转置为 (Time, Vars, H, W)
            patch_val = patch_val.transpose(1, 0, 2, 3)
            
            # 填入大数组
            all_data[i] = patch_val
            success_count += 1
            
        except Exception as e:
            print(f"⚠️ 样本 {i} 读取失败: {e}")
            continue

    print(f"💾 正在保存 {success_count} 个样本...")
    
    new_ds = xr.Dataset(
        {
            "data": (["sample", "time", "variable", "y", "x"], all_data[:success_count])
        },
        coords={
            "sample": np.arange(success_count),
            "variable": FEATURE_NAMES,
        }
    )
    
    # 压缩保存
    encoding = {'data': {'zlib': True, 'complevel': 5}}
    new_ds.to_netcdf(OUTPUT_NC_PATH, engine='h5netcdf', encoding=encoding)
    
    print(f"✅ 搞定！文件已保存至: {OUTPUT_NC_PATH}")

extract_surgical()