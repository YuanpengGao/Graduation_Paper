import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
import numpy as np
import os
import time
import xarray as xr
import swanlab
import random

# 引入你自己写的文件 (确保文件名对得上)
from dataset.dataset_loader import FireDataset  
from model.model_ConvLSTM import ConvLSTMModel     
from model.loss import DiceLoss, FocalLoss, TverskyLoss, LovaszHingeLoss

# 超参数
BATCH_SIZE = 32        # 显存够大可以改大，比如 32
LEARNING_RATE = 5e-4   # 学习率，ConvLSTM 建议小一点
NUM_EPOCHS = 200        # 训练轮数
SEQ_LEN = 10            # 输入过去 10 天
BCE_WEIGHT = 1.0
FOCAL_WEIGHT = 1.0
DATA_PATH = './data/mixed_1_3_train_data.nc' # 数据路径
SAVE_PATH = './checkpoint/best_model_mixed.pth'       # 模型保存路径
POS_WEIGHT = 2.0
Tversky_Beta = 0.7
Tversky_Alpha = 0.3

swanlab.init(
  # 设置将记录此次运行的项目信息
  project="AIforFireSpread",
  workspace="Micro",
  # 跟踪超参数和运行元数据
  config={
    "learning_rate": LEARNING_RATE,
    "architecture": "ConvLSTM",
    "dataset": "FireCube",
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "seq_len": SEQ_LEN,
    "bce_weight": BCE_WEIGHT,
    "pos_weight": POS_WEIGHT
  }
)
# ================= 配置参数区 =================
# 显卡设置
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda") # NVIDIA 显卡
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # 🍎 Mac M1/M2/M3... 显卡
    else:
        return torch.device("cpu")  # 只有 CPU

DEVICE = get_device()
print(f"🚀 当前使用的计算设备: {DEVICE}")


def get_dataloaders():
    print(f"🚀 正在加载预处理好的数据集: {DATA_PATH} ...")
    
    # 1. 实例化 Dataset
    # 注意：现在的 Dataset 很简单，只需要文件路径，不需要 seq_len 等复杂参数
    # 因为数据在制作时已经切好了
    full_dataset = FireDataset(DATA_PATH)
    
    total_len = len(full_dataset)
    print(f"📊 数据集总样本数: {total_len}")
    
    # 2. 划分 训练/验证/测试 (8:1:1)
    # 因为这是独立切片的数据，可以使用 random_split
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    
    # 使用 random_split 随机打乱划分
    train_set, val_set, test_set = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # 固定随机种子，保证每次划分一样
    )
    
    print(f"   - 训练集: {len(train_set)}")
    print(f"   - 验证集: {len(val_set)}")
    print(f"   - 测试集: {len(test_set)}")
    
    # 3. 创建 DataLoader
    # 训练集 shuffle=True，其他 False
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

# ================= 2. 训练与验证函数 =================
def train_one_epoch(model, loader, criterion_bce, criterion_dice, criterion_focal, optimizer):
    model.train() # 开启训练模式
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        # 搬运数据到显卡
        # inputs: (B, T, C, H, W) -> float32
        # targets: (B, 1, H, W)   -> float32
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)

                
        # 1. 梯度清零
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(inputs) # outputs 也是 (B, 1, H, W)
        
        # 3. 计算 Loss
        # 注意：BCEWithLogitsLoss 内部自带 Sigmoid，所以模型输出不需要加 Sigmoid
        loss_bce = criterion_bce(outputs, targets)
        loss_dice = criterion_dice(outputs, targets)
        loss_focal = criterion_focal(outputs, targets)
        
        
        loss = BCE_WEIGHT * loss_bce + loss_dice + loss_focal * FOCAL_WEIGHT
        
        # 4. 反向传播与优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"   Step [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f} BCE Loss: {loss_bce.item():.4f} DICE_Loss: {loss_dice.item():.4f}  Focal Loss: {loss_focal.item():.4f}")
        
        swanlab.log(
            {"Train/Loss": loss.item(),
            "Train/BCELoss": loss_bce.item(), 
            "Train/DICE_Loss": loss_dice.item(),
            "Train/Focal_Loss": loss_focal.item()})
            
    return running_loss / len(loader)

def validate(model, loader, criterion_bce, criterion_dice, criterion_focal):
    model.eval() # 开启评估模式 (关闭 Dropout 等)
    running_loss = 0.0
    
    with torch.no_grad(): # 不计算梯度，节省显存
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            
            outputs = model(inputs)
            loss_bce = criterion_bce(outputs, targets)
            loss_dice = criterion_dice(outputs, targets)
            loss_focal = criterion_focal(outputs, targets)
            loss = BCE_WEIGHT * loss_bce + loss_dice + loss_focal * FOCAL_WEIGHT
            running_loss += loss.item()
            
    return running_loss / len(loader)

# ================= Main 函数 =================
if __name__ == "__main__":
    # 1. 获取数据
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 2. 初始化模型
    # 假设你的输入有 5 个通道 (Fire, NDVI, Temp, Wind, FWI)
    model = ConvLSTMModel(input_channels=8, hidden_channels=16).to(DEVICE)
    
    # 3. 定义 Loss 和 优化器
    # 🚨 关键点：加权 Loss
    # 因为火灾像素很少(0少1多)，我们要给火灾像素(1)更高的权重，强迫模型关注火灾
    
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)) # 权重还是给大点
    criterion_dice = TverskyLoss(alpha=Tversky_Alpha, beta=Tversky_Beta)
    criterion_focal = FocalLoss()
    criterion_dice = LovaszHingeLoss(per_image=True).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 4. 开始训练循环
    best_val_loss = float('inf')
    
    print("\n🔥 开始训练...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion_bce, criterion_dice, criterion_focal, optimizer)
        
        # 验证
        val_loss = validate(model, val_loader, criterion_bce, criterion_dice, criterion_focal)
        swanlab.log({
            "Val/Loss": val_loss,
            "epoch": epoch + 1  # 推荐：把 epoch 也记下来，方便对齐
        })
        
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            print(f"   ✅发现更优模型 (Loss: {best_val_loss:.4f} -> {val_loss:.4f})，已保存！")
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            print(f"   (未提升，最佳 Loss: {best_val_loss:.4f})")
            
    total_time = time.time() - start_time
    print(f"\n✨ 训练结束！总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳模型已保存在: {SAVE_PATH}")

    swanlab.finish()

    # ================= 5. 最终测试 (新增部分) =================
    print("\n🔍 正在加载最佳模型进行最终测试...")
    
    # 1. 重新加载保存的“最佳模型”参数
    # (这一步很重要，因为现在的 model 变量是训练到最后一步的模型，不一定是最好的)
    best_state = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(best_state)
    
    # 2. 在测试集上跑一遍
    # (直接复用 validate 函数就行，因为逻辑一样：只算 Loss，不更新参数)
    test_loss = validate(model, test_loader, criterion_bce, criterion_dice, criterion_focal)
    
    print("="*40)
    print(f"🏆 最终测试集 Loss: {test_loss:.4f}")
    print("="*40)
    
    if test_loss < 0.6: # 这里的阈值只是举例
        print("✨ 结果看起来不错！模型具备泛化能力。")
    else:
        print("⚠️ Loss 偏高，可能需要调整模型结构或增加数据。")