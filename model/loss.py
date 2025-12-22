import torch.nn as nn
import torch
import torch.nn.functional as F

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float): 控制正负样本权重的参数 (0 < alpha < 1)。
                           对于正样本较少的情况，通常设置 alpha > 0.5 (如 0.75) 来增加正样本权重。
                           或者配合 gamma 使用时，alpha 通常设为 0.25 (保留原始论文设置)。
            gamma (float): 聚焦参数，控制对困难样本的挖掘程度。gamma 越大，越关注难分的样本。
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: 模型的原始输出 (Logits), 未经过 Sigmoid
        # targets: 真实标签 (0 或 1)
        
        # 1. 计算二元交叉熵 (BCE)
        # reduction='none' 保证我们可以对每个像素单独加权
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 2. 计算 p_t (模型预测该类别的概率)
        # 如果 target=1, pt = p; 如果 target=0, pt = 1-p
        pt = torch.exp(-bce_loss) 
        
        # 3. 计算 Focal Loss 公式: -alpha * (1-pt)^gamma * log(pt)
        # alpha_t: 如果 target=1 则为 alpha, 否则为 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.8, smooth=1e-6):
        """
        alpha + beta = 1
        beta 越大，对 False Negative (漏报) 的惩罚越重。
        如果模型一直预测全黑，尝试把 beta 设为 0.8 甚至 0.9。
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: Logits (未经过 Sigmoid)
        inputs = torch.sigmoid(inputs)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算 TP, FP, FN
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - Tversky
    
class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True, ignore=None):
        """
        二分类 Lovasz Hinge Loss
        Args:
            per_image (bool): 是否对每个图像单独计算 Loss 然后平均。
                              对于 Batch Size 较小的情况，建议设为 True。
            ignore (int): 需要忽略的类别 ID (通常不需要，除非有 mask 掉的区域)
        """
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, 1, H, W] 模型的原始输出 (Logits)，未经过 Sigmoid！
            labels: [B, 1, H, W] 真实标签 (0 或 1)
        """
        if self.per_image:
            loss = torch.mean(torch.stack([
                self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0)))
                for log, lab in zip(logits, labels)
            ]))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels))
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        计算打平后的 Hinge Loss
        """
        if len(labels) == 0:
            # 只有忽略区域，Loss 为 0
            return logits.sum() * 0.
        
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def lovasz_grad(self, gt_sorted):
        """
        计算 Lovasz 扩展的梯度
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # 处理 1 像素的情况
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def flatten_binary_scores(self, scores, labels):
        scores = scores.view(-1)
        labels = labels.view(-1)
        if self.ignore is None:
            return scores, labels
        valid = (labels != self.ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels