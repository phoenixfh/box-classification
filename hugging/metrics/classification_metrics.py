"""
分类任务的metrics计算
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    计算分类metrics
    
    Args:
        eval_pred: EvalPrediction对象，包含predictions和labels
        
    Returns:
        metrics字典
    """
    predictions, labels = eval_pred
    
    # 获取预测类别
    if len(predictions.shape) > 1:
        # logits: [batch_size, num_classes]
        preds = np.argmax(predictions, axis=1)
    else:
        # 已经是类别ID
        preds = predictions
    
    # 计算accuracy
    accuracy = accuracy_score(labels, preds)
    
    # 计算precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
