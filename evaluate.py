import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# 引入你的模块
from dataset.dataset_loader import FireDataset
from model.model_ConvLSTM import ConvLSTMModel

# ================= 配置区 =================
# 确保这里指向你最新的混合数据集
DATA_PATH = './data/balanced_train_data.nc' 
MODEL_PATH = './checkpoint/best_model.pth'

INPUT_CHANNELS = 8
HIDDEN_CHANNELS = 32
BATCH_SIZE = 16
THRESHOLD = 0.5  # 判定阈值：概率 > 0.5 算有火

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================

def calculate_batch_metrics(pred_bin, target):
    """
    计算一个 Batch 的指标
    pred_bin: (B, 1, H, W) -> 0或1
    target:   (B, 1, H, W) -> 0或1
    """
    smooth = 1e-6
    
    # 展平以便计算 (B, -1)
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Intersection (交集)
    intersection = (pred_flat * target_flat).sum(dim=1)
    
    # Union (并集)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    # 1. IoU Calculation
    # 如果 Union 是 0 (即预测和真实全黑)，IoU 应该算 1 (预测完美)
    iou = (intersection + smooth) / (union + smooth)
    # 修正：对于完全无火且预测也无火的情况，IoU=1
    no_fire_mask = (union == 0)
    iou[no_fire_mask] = 1.0
    
    # 2. Recall Calculation (TP / (TP + FN))
    # 真实有火的地方，预测对了多少？
    target_positives = target_flat.sum(dim=1)
    recall = (intersection + smooth) / (target_positives + smooth)
    # 修正：如果真实图里根本没火，Recall 这一项通常不计入或算1
    # 这里我们定义：如果没火，Recall = 1 (没有漏报)
    no_target_mask = (target_positives == 0)
    recall[no_target_mask] = 1.0
    
    # 3. Precision Calculation (TP / (TP + FP))
    # 预测有火的地方，真的有火吗？
    pred_positives = pred_flat.sum(dim=1)
    precision = (intersection + smooth) / (pred_positives + smooth)
    # 修正：如果预测全黑，Precision = 1
    no_pred_mask = (pred_positives == 0)
    precision[no_pred_mask] = 1.0
    
    return iou.mean().item(), recall.mean().item(), precision.mean().item()

def main():
    print(f"🚀 开始评估模型: {MODEL_PATH}")
    
    # 1. 加载模型
    model = ConvLSTMModel(INPUT_CHANNELS, HIDDEN_CHANNELS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print("✅ 模型权重加载成功")
    else:
        print("❌ 模型文件不存在！")
        return
    model.eval()

    # 2. 准备测试集
    # 注意：这里的划分种子 (seed=42) 必须和 train.py 里一模一样！
    # 否则你的测试集可能会混入训练集的数据 (数据泄露)
    print("📥 加载数据集...")
    full_dataset = FireDataset(DATA_PATH)
    total_len = len(full_dataset)
    
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    
    generator = torch.Generator().manual_seed(42) # 必须和训练时一致
    _, _, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"📊 测试集样本数: {len(test_set)}")

    # 3. 开始推理
    iou_list = []
    recall_list = []
    precision_list = []
    
    print("🔍 正在计算指标...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            
            # 推理
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            # 二值化 (Thresholding)
            preds_bin = (probs > THRESHOLD).float()
            
            # 计算当前 Batch 指标
            iou, recall, precision = calculate_batch_metrics(preds_bin, targets)
            
            iou_list.append(iou)
            recall_list.append(recall)
            precision_list.append(precision)

    # 4. 打印最终结果
    avg_iou = np.mean(iou_list)
    avg_recall = np.mean(recall_list)
    avg_precision = np.mean(precision_list)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)

    print("\n" + "="*40)
    print(f"📈 测试集评估报告 (Threshold={THRESHOLD})")
    print("="*40)
    print(f"IoU (交并比)      : {avg_iou:.4f}")
    print(f"Recall (召回率)   : {avg_recall:.4f}")
    print(f"Precision (精确率): {avg_precision:.4f}")
    print(f"F1 Score         : {f1_score:.4f}")
    print("="*40)
    
    # 简单的结果解读
    print("\n💡 结果解读建议:")
    if avg_iou > 0.4:
        print("✅ IoU > 0.4: 对于野火这种极难的小目标分割，这已经是很不错的成绩了！")
    else:
        print("⚠️ IoU 较低: 说明预测的火灾形状和真实形状重合度不高。")
        
    if avg_recall > 0.7:
        print("✅ Recall 高: 说明模型很灵敏，很少漏报火灾（这对防灾很重要）。")
    elif avg_recall < 0.5:
        print("⚠️ Recall 低: 模型经常漏报，可能需要调低阈值 (Threshold)。")
        
    if avg_precision < 0.3:
        print("⚠️ Precision 低: 模型可能在“乱猜”，误报率很高（即使没火也预测有火）。")

if __name__ == "__main__":
    main()