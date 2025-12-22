# 毕业论文代码目录
## 先写给自己的数据目录
/data目录下的数据切分的具体含义
- dataset_greece.nc 原始的FireCube数据集
- mini_dataset.nc 切分的2021年的128*128的样本，共约240个，数据集较小
- dataset_allyear.nc 全部年份的某块128*128的数据集
- balanced_train_data.nc 采样的五百个正样本，也是128*128
- mixed_1_3_train_data.nc 正负样本1：3，实际为1000：3000，地理上也是128*128

npy文件
- fire_indices.npy 储存了所有正样本的索引

## checkpoint目录
- best_model_balanced.pth 针对balanced_train_data.nc训练的最好模型权重
- best_model_mixed.pth 针对mixed_1_3_train_data.nc训练的最好模型权重