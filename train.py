import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import time

# 引入你自己写的文件 (确保文件名对得上)
from dataset.dataset_loader import FireDataset  
from model.model_ConvLSTM import ConvLSTMModel     
from model.loss import DiceLoss  

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

# 超参数
BATCH_SIZE = 16        # 显存够大可以改大，比如 32
LEARNING_RATE = 1e-4   # 学习率，ConvLSTM 建议小一点
NUM_EPOCHS = 200        # 训练轮数
SEQ_LEN = 7            # 输入过去 7 天
DATA_PATH = './data/mini_dataset.nc' # 数据路径
SAVE_PATH = './checkpoint/best_model.pth'       # 模型保存路径

# ================= 1. 准备数据 =================
def get_dataloaders():
    print("正在加载数据集...")
    # 初始化你的 Dataset
    full_dataset = FireDataset(DATA_PATH, seq_len=SEQ_LEN)
    total_len = len(full_dataset)
    
    # --- 关键步骤：按时间切分 (7:1:2) ---
    train_size = int(0.7 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    
    # 生成不打乱的索引
    indices = list(range(total_len))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    # 创建子集
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)
    
    print(f"📊 数据划分完成:")
    print(f"   - 训练集: {len(train_set)} 样本")
    print(f"   - 验证集: {len(val_set)} 样本")
    print(f"   - 测试集: {len(test_set)} 样本")
    
    # 创建 DataLoader
    # 注意：训练集通常 shuffle=True (打乱样本顺序，但不打乱时间步内部顺序)，防止模型死记硬背
    # 验证和测试集 shuffle=False
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# ================= 2. 训练与验证函数 =================
def train_one_epoch(model, loader, criterion_bce, criterion_dice, optimizer):
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
        loss = loss_bce + loss_dice
        
        # 4. 反向传播与优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"   Step [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")
            
    return running_loss / len(loader)

def validate(model, loader, criterion_bce, criterion_dice):
    model.eval() # 开启评估模式 (关闭 Dropout 等)
    running_loss = 0.0
    
    with torch.no_grad(): # 不计算梯度，节省显存
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            
            outputs = model(inputs)
            loss_bce = criterion_bce(outputs, targets)
            loss_dice = criterion_dice(outputs, targets)
            loss = loss_bce + loss_dice
            running_loss += loss.item()
            
    return running_loss / len(loader)

# ================= Main 函数 =================
if __name__ == "__main__":
    # 1. 获取数据
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 2. 初始化模型
    # 假设你的输入有 5 个通道 (Fire, NDVI, Temp, Wind, FWI)
    model = ConvLSTMModel(input_channels=5, hidden_channels=32).to(DEVICE)
    
    # 3. 定义 Loss 和 优化器
    # 🚨 关键点：加权 Loss
    # 因为火灾像素很少(0少1多)，我们要给火灾像素(1)更高的权重，强迫模型关注火灾
    
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]).to(DEVICE)) # 权重还是给大点
    criterion_dice = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. 开始训练循环
    best_val_loss = float('inf')
    
    print("\n🔥 开始训练...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion_bce, criterion_dice, optimizer)
        
        # 验证
        val_loss = validate(model, val_loader, criterion_bce, criterion_dice,)
        
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

    # ================= 5. 最终测试 (新增部分) =================
    print("\n🔍 正在加载最佳模型进行最终测试...")
    
    # 1. 重新加载保存的“最佳模型”参数
    # (这一步很重要，因为现在的 model 变量是训练到最后一步的模型，不一定是最好的)
    best_state = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(best_state)
    
    # 2. 在测试集上跑一遍
    # (直接复用 validate 函数就行，因为逻辑一样：只算 Loss，不更新参数)
    test_loss = validate(model, test_loader, criterion_bce, criterion_dice,)
    
    print("="*40)
    print(f"🏆 最终测试集 Loss: {test_loss:.4f}")
    print("="*40)
    
    if test_loss < 0.6: # 这里的阈值只是举例
        print("✨ 结果看起来不错！模型具备泛化能力。")
    else:
        print("⚠️ Loss 偏高，可能需要调整模型结构或增加数据。")