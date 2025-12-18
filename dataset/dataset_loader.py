import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

class FireDataset(Dataset):
    def __init__(self, nc_file, seq_len=7, target_col="burned_areas"):
        """
        Args:
        nc_file: nc格式数据集的路径
        seq_len: 滑动窗口的数据长度
        target_col: 需要预测的列，此处为燃烧面积
        """
        super().__init__()
        self.seq_len = seq_len

        ds = xr.open_dataset(nc_file)

        # 定义我们需要的特征
        self.feature_names = [
            'burned_areas','ndvi',
            'fwi','max_t2m','max_wind_speed']
        
        # 把数据集转化成array,全部加载到内存
        data_array = ds[self.feature_names].to_array(dim='variable')
        self.data = data_array.transpose('time','variable','y','x').values

        # 找到需要预测的变量的位置
        self.target_idx = self.feature_names.index(target_col)

        self.data = self.data.astype(np.float32)
        
        print("数据集加载完成。")
        print(f"数据形状:{self.data.shape}(天数，特征数，高，宽)")
        print(f"特征列表:{self.feature_names}")

        for i, name in enumerate(self.feature_names):
            if np.isnan(self.data[:,i,:,:]).any():
                print(f"\n注意!变量{name}中含有NaN值。")
            else:
                print(f"变量{name}中不含有NaN值。")


    def __len__(self):
        '''
        总长度减去滑动窗口的序列长度
        '''
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        '''
        取第idx个样本,为训练输入做准备
        输入X:第idx开始的seq_len长度的样本
        输出Y:第idx+seq_len个样本
        '''
        x = self.data[idx:idx+self.seq_len]

        target_day_data = self.data[idx+self.seq_len]
        y = target_day_data[self.target_idx]
        y = y[np.newaxis,...]

        return torch.from_numpy(x), torch.from_numpy(y)
        
if __name__ == "__main__":
    dataset = FireDataset("./data/mini_dataset.nc",seq_len=7)
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    
    sample_x, sample_y = next(iter(loader))
    print("\n-------维度检查-------")
    print(f"Input Batch Shape(X):{sample_x.shape}")
    print("格式应为: (Batch, Time, Features, Height, Width)")
    print(f"Output Batch Shape(Y):{sample_y.shape}")
    print("格式应为: (Batch, 1, Height, Width)")

    if torch.isnan(sample_x).any():
        print("警告:输入数据中包含NAN值!")
    else:
        print("数据检查通过,无NAN值。")


