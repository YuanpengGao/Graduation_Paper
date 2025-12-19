import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 引入你的模型和数据集类
from dataset.dataset_loader import FireDataset
from model.model_ConvLSTM import ConvLSTMModel

# ================= 配置 =================
# 必须和训练时保持一致
SEQ_LEN = 7
INPUT_CHANNELS = 5
HIDDEN_CHANNELS = 32
DATA_PATH = './data/mini_dataset.nc'
MODEL_PATH = './checkpoint/best_model.pth'

# 设备选择 (自动适配 Mac)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"🚀 推理设备: {DEVICE}")

def load_model():
    print("📥 正在加载模型权重...")
    model = ConvLSTMModel(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS)
    
    # 关键：map_location 确保在 CPU/Mac 上也能加载 CUDA 训练的模型
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval() # 开启评估模式 (关闭 Dropout 等)
    print("✅ 模型加载成功！")
    return model

def visualize_result(input_seq, target, prediction, save_name="result.png"):
    """
    input_seq: (Time, C, H, W) - 输入序列
    target: (1, H, W) - 真实标签
    prediction: (1, H, W) - 模型预测概率
    """
    
    # 取出 "昨天" (序列的最后一天) 的火灾情况作为对比
    # 假设 Channel 0 是 Burned Areas
    last_input_fire = input_seq[-1, 0, :, :] 
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 昨天的火 (Input T-1)
    ax = axes[0]
    im = ax.imshow(last_input_fire, cmap='Reds', vmin=0, vmax=1)
    ax.set_title("Input: Yesterday's Fire")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 2. 今天的真火 (Ground Truth)
    ax = axes[1]
    im = ax.imshow(target[0], cmap='Reds', vmin=0, vmax=1)
    ax.set_title("Target: Today's Real Fire")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 3. AI 预测的概率 (Prediction Probability)
    ax = axes[2]
    # 使用 inferno 色阶，越亮代表概率越大
    im = ax.imshow(prediction[0], cmap='inferno', vmin=0, vmax=1)
    ax.set_title("AI Prediction (Probability)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 4. 二值化后的预测 (Binary Prediction > 0.5)
    # 看看如果按 0.5 切一刀，AI 到底认为哪里着火了
    ax = axes[3]
    binary_pred = (prediction[0] > 0.3).astype(float) # 阈值设低一点(0.3)更容易看清
    im = ax.imshow(binary_pred, cmap='Reds', vmin=0, vmax=1)
    ax.set_title("AI Decision (Threshold > 0.3)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"✨ 图片已保存为: {save_name} (请在左侧文件栏打开查看)")
    plt.show() # 如果在 Mac 本地跑，会直接弹窗显示

def main():
    # 1. 准备模型
    model = load_model()
    
    # 2. 准备数据 (只取测试集部分)
    print("📥 正在加载数据集...")
    full_dataset = FireDataset(DATA_PATH, seq_len=SEQ_LEN)
    total_len = len(full_dataset)
    test_start_idx = int(0.8 * total_len) # 假设后20%是测试集
    
    # 3. 🔍 寻找“连续火灾”样本
    # 我们不再随机抽，而是遍历测试集，直到找到一个完美的样本
    print(f"🔍 正在遍历测试集 (索引 {test_start_idx} -> {total_len}) 寻找【连续火灾】样本...")
    
    found_idx = -1
    found_inputs = None
    found_target = None
    
    # 遍历测试集
    for i in range(test_start_idx, total_len):
        inputs, target = full_dataset[i]
        
        # inputs shape: (7, 5, 128, 128)
        # target shape: (1, 128, 128)
        
        # 假设 Channel 0 是 burned_areas
        # 1. 检查昨天 (序列最后一天) 是否有火
        yesterday_fire = inputs[-1, 0, :, :]
        has_fire_yesterday = yesterday_fire.max() > 0
        
        # 2. 检查今天 (Target) 是否有火
        has_fire_today = target.max() > 0
        
        # 🔥 核心条件：昨天有火 AND 今天也有火
        if has_fire_yesterday and has_fire_today:
            print(f"🎯 找到了！索引: {i}")
            print(f"   - 昨天火点像素数: {(yesterday_fire > 0).sum()}")
            print(f"   - 今天火点像素数: {(target > 0).sum()}")
            
            found_idx = i
            found_inputs = inputs
            found_target = target
            break # 找到一个就停，想多找几个可以注释掉 break
    
    if found_idx == -1:
        print("⚠️ 哎呀，测试集里没找到【连续两天都有火】的样本。")
        print("   -> 可能是测试集刚好切到了没火的时间段，或者火灾刚好在昨天熄灭了。")
        print("   -> 正在随机抽取一个样本兜底...")
        rand_idx = random.randint(test_start_idx, total_len - 1)
        found_inputs, found_target = full_dataset[rand_idx]
    
    # 4. 推理
    # 增加 Batch 维度: (T, C, H, W) -> (1, T, C, H, W)
    input_tensor = found_inputs.unsqueeze(0).to(DEVICE, dtype=torch.float)
    
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.sigmoid(logits)
        
    # 转回 CPU numpy 方便画图
    inputs_np = found_inputs.numpy()
    target_np = found_target.numpy()
    preds_np = preds.cpu().numpy()[0] 
    
    # 5. 可视化
    print(f"🎨 正在绘制索引 {found_idx if found_idx != -1 else rand_idx} 的预测结果...")
    visualize_result(inputs_np, target_np, preds_np, save_name="prediction_continuous_fire.png")
    
if __name__ == "__main__":
    main()