import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 引入你的模型和数据集类
# ⚠️ 确保 dataset_loader.py 已经是更新后的极简版本
from dataset.dataset_loader import FireDataset
from model.model_ConvLSTM import ConvLSTMModel

# ================= 配置 =================
# 必须和训练时保持一致
INPUT_CHANNELS = 8
HIDDEN_CHANNELS = 32

# 指向你刚才生成的 balanced 数据集
DATA_PATH = './data/balanced_train_data.nc' 
# 指向训练好的最佳模型
MODEL_PATH = './checkpoint/best_model_balanced.pth'

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
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval() # 开启评估模式
        print("✅ 模型加载成功！")
        
        # 打印 Bias 看看有没有被训练坏
        print(f"DEBUG - Output Layer Bias: {model.output_layer.bias.data.cpu().numpy()}")
        return model
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        return None

def visualize_result(input_seq, target, prediction, save_name="result.png"):
    """
    input_seq: (Time, C, H, W) - 输入序列 (经过归一化的)
    target: (1, H, W) - 真实标签
    prediction: (1, H, W) - 模型预测概率
    """
    
    # 取出 "昨天" (序列的最后一天) 的火灾情况作为对比
    # 假设 Channel 0 是 Burned Areas
    # 注意：这里的 input_seq 是归一化过的，数值可能不是 0/1
    # 但形状还是看得到的
    last_input_fire = input_seq[-1, 0, :, :] 
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 昨天的火 (Input T-1)
    ax = axes[0]
    im = ax.imshow(last_input_fire, cmap='Reds') # 去掉 vmin/vmax 以便观察归一化后的值
    ax.set_title("Input: Yesterday's Fire (Normalized)")
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
    ax = axes[3]
    # 阈值设为 0.5 看看效果
    binary_pred = (prediction[0] > 0.5).astype(float) 
    im = ax.imshow(binary_pred, cmap='Reds', vmin=0, vmax=1)
    ax.set_title("AI Decision (Threshold > 0.5)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"✨ 图片已保存为: {save_name}")
    # plt.show() # 如果在服务器上跑，这一行可以注释掉

def main():
    # 1. 准备模型
    model = load_model()
    if model is None: return
    
    # 2. 准备数据
    print(f"📥 正在加载数据集: {DATA_PATH} ...")
    
    # 🔥【修改点】不再需要传 seq_len，因为 dataset 内部已经切好了
    try:
        full_dataset = FireDataset(DATA_PATH)
    except TypeError:
        print("❌ 错误：Dataset 初始化失败。请检查 dataset_loader.py 是否已更新为不需要 seq_len 的版本。")
        return

    total_len = len(full_dataset)
    print(f"📊 数据集总数: {total_len}")
    
    # 3. 寻找测试样本
    # 因为现在全是正样本，所以我们不需要太复杂的搜索逻辑，随便拿几个都应该有火
    print(f"🔍 正在抽取样本进行测试...")
    
    found_idx = -1
    found_inputs = None
    found_target = None
    
    # 简单遍历前 20 个样本，找一个火点比较明显的
    for i in range(min(20, total_len)):
        inputs, target = full_dataset[i]
        
        # 统计火点像素
        fire_pixels = (target > 0).sum()
        
        if fire_pixels > 10: # 找一个火点超过10个像素的，画出来比较明显
            print(f"🎯 找到优质样本！索引: {i}, 火点像素数: {fire_pixels}")
            found_idx = i
            found_inputs = inputs
            found_target = target
            break
            
    if found_idx == -1:
        print("⚠️ 没找到火点很多的样本，随机取一个...")
        found_idx = 0
        found_inputs, found_target = full_dataset[0]

    # 4. 推理
    # 增加 Batch 维度: (T, C, H, W) -> (1, T, C, H, W)
    input_tensor = found_inputs.unsqueeze(0).to(DEVICE, dtype=torch.float)
    
    print("\n------- Debug 信息 -------")
    print(f"Input Min: {input_tensor.min().item():.4f}, Max: {input_tensor.max().item():.4f}")
    if torch.isnan(input_tensor).any():
        print("❌ 警告：输入数据包含 NaN！")
    
    with torch.no_grad():
        logits = model(input_tensor)
        
        print(f"Logits Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
        print(f"Logits Mean: {logits.mean().item():.4f}")
        
        preds = torch.sigmoid(logits)
        print(f"Preds Max Prob: {preds.max().item():.4f}")
        
    # 转回 CPU numpy 方便画图
    inputs_np = found_inputs.numpy()
    target_np = found_target.numpy()
    preds_np = preds.cpu().numpy()[0] 
    
    # 5. 可视化
    visualize_result(inputs_np, target_np, preds_np, save_name="prediction_balanced.png")
    
if __name__ == "__main__":
    main()