"""
优化器工厂

创建不同类型的优化器
"""

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from typing import Union


def create_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        optimizer_name: 优化器名称 ('SGD', 'Adam', 'AdamW', 'RMSprop')
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他优化器参数
        
    Returns:
        优化器实例
    """
    # 获取模型参数
    params = model.parameters()
    
    if optimizer_name == 'SGD':
        optimizer = SGD(
            params,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_name == 'Adam':
        optimizer = Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name == 'AdamW':
        optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8)
        )
    
    else:
        raise ValueError(
            f"不支持的优化器: {optimizer_name}\n"
            f"支持的优化器: SGD, Adam, AdamW, RMSprop"
        )
    
    return optimizer
