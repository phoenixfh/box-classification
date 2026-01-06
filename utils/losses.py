"""
损失函数
"""

import torch
import torch.nn as nn

from .metrics import dice_score


def dice_loss(preds, targets, smooth=1e-6):
    """
    Dice Loss（用于分割任务）
    
    Args:
        preds: 预测值 tensor
        targets: 目标值 tensor
        smooth: 平滑系数
        
    Returns:
        torch.Tensor: Dice loss (1 - dice_score)
    """
    return 1 - dice_score(preds, targets, smooth)


class CombinedLoss:
    """
    组合损失: BCE + Dice
    
    用于分割任务，结合二元交叉熵和 Dice 损失
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """
        Args:
            bce_weight: BCE 损失权重
            dice_weight: Dice 损失权重
        """
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def __call__(self, preds, targets):
        """
        计算组合损失
        
        Args:
            preds: 预测值 [B, 1, H, W] 或 [B, H, W]
            targets: 目标值 [B, H, W]
            
        Returns:
            torch.Tensor: 组合损失
        """
        targets = targets.float()
        
        # 统一处理形状
        if preds.dim() == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        bce = self.bce(preds, targets)
        dice = dice_loss(preds, targets)
        return self.bce_weight * bce + self.dice_weight * dice
