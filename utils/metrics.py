"""
评估指标
"""

import torch
from fastai.metrics import Metric


def dice_score(preds, targets, smooth=1e-6):
    """
    计算 Dice 系数（用于分割任务）
    
    Args:
        preds: 预测值 tensor [B, 1, H, W] 或 [B, H, W]
        targets: 目标值 tensor [B, H, W]
        smooth: 平滑系数，避免除零
        
    Returns:
        torch.Tensor: Dice 系数
    """
    # 处理预测值
    if preds.dim() == 4:  # [B, 1, H, W]
        preds = preds.squeeze(1)  # -> [B, H, W]
    
    # 应用sigmoid得到概率
    preds = preds.sigmoid()
    
    # 展平
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1).float()
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


class DiceMetric(Metric):
    """Dice 评估指标（FastAI Metric）"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total = 0.
        self.count = 0
    
    def accumulate(self, learn):
        preds = learn.pred
        targets = learn.y
        self.total += dice_score(preds, targets).item()
        self.count += 1
    
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0.
