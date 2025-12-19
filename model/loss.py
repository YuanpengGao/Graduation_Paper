import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. 把 Logits 转成概率 (0~1)
        inputs = torch.sigmoid(logits)
        
        # 2. 展平 (Batch, Time, H, W) -> (N,)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 3. 计算交集 (Intersection)
        intersection = (inputs * targets).sum()
        
        # 4. Dice 系数公式: 2*交集 / (预测总面积 + 真实总面积)
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # 5. Loss = 1 - Dice
        return 1 - dice