"""
分类模型

提供简化的模型创建接口
"""

import torch.nn as nn
from .model_factory import ModelFactory


def create_classification_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True
) -> nn.Module:
    """
    创建图像分类模型
    
    Args:
        arch: 模型架构名称
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        
    Returns:
        分类模型
    
    Examples:
        >>> model = create_classification_model('resnet18', num_classes=10)
        >>> model = create_classification_model('vit_base', num_classes=100, pretrained=True)
    """
    return ModelFactory.create_model(arch, num_classes, pretrained)
