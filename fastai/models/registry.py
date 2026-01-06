"""模型注册表 - 管理所有自定义模型"""
from typing import Dict, Callable
import torch.nn as nn

# 全局模型注册表
_MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """
    模型注册装饰器
    
    使用示例:
        @register_model('unet')
        def create_unet(n_classes, **kwargs):
            return UNet(n_channels=3, n_classes=n_classes)
    
    Args:
        name: 模型名称（不区分大小写）
    """
    def decorator(func: Callable):
        _MODEL_REGISTRY[name.lower()] = func
        return func
    return decorator

def get_model(name: str, n_classes: int, **kwargs) -> nn.Module:
    """
    获取注册的模型实例
    
    Args:
        name: 模型名称
        n_classes: 输出类别数
        **kwargs: 模型特定参数（如img_size, bilinear等）
    
    Returns:
        PyTorch模型实例
    
    Raises:
        ValueError: 模型未注册
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"❌ 模型 '{name}' 未注册！\n"
            f"   可用的自定义模型: {available}\n"
            f"   或使用fastai内置模型: resnet18, resnet50, efficientnet_b0 等"
        )
    return _MODEL_REGISTRY[name_lower](n_classes=n_classes, **kwargs)

def list_models() -> list:
    """列出所有已注册的自定义模型"""
    return sorted(_MODEL_REGISTRY.keys())

def is_custom_model(name: str) -> bool:
    """检查是否为已注册的自定义模型"""
    return name.lower() in _MODEL_REGISTRY
