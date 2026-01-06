"""UNet模型 - 从Pytorch-UNet复制并适配"""
import torch.nn as nn
from .unet_model import UNet
from ..registry import register_model

# 变体定义
VARIANTS = {
    'n': 'nano',
    's': 'small', 
    'm': 'medium',
    'l': 'large',
    'x': 'xlarge'
}

# 自动注册所有分割模型变体
for variant_key, variant_name in VARIANTS.items():
    # 使用闭包捕获变量
    def _make_seg_creator(var_key):
        @register_model(f'unet_{var_key}_seg')
        def _create_seg(n_classes: int, n_channels: int = 3, bilinear: bool = False, **kwargs):
            f"""UNet {variant_name} segmentation model"""
            return UNet(n_channels, n_classes, bilinear, variant=var_key)
        return _create_seg
    
    def _make_cls_creator(var_key):
        @register_model(f'unet_{var_key}')
        def _create_cls(n_classes: int, n_channels: int = 3, bilinear: bool = False, **kwargs):
            f"""UNet {variant_name} classification model"""
            class UNetClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = UNet(n_channels, n_classes, bilinear, variant=var_key)
                def forward(self, x):
                    seg_map = self.unet(x)
                    return seg_map.mean(dim=(-2, -1))
            return UNetClassifier()
        return _create_cls
    
    # 调用工厂函数创建注册器
    _make_seg_creator(variant_key)
    _make_cls_creator(variant_key)

# 保持向后兼容 - unet_seg 默认使用 small 变体
@register_model('unet_seg')
def create_unet_segmentation(n_classes: int, n_channels: int = 3, 
                             bilinear: bool = False, **kwargs):
    """
    UNet分割模型（默认使用 small 变体）
    - 输出: [batch_size, n_classes, H, W] 的分割图
    
    Args:
        n_classes: 输出类别数（分割类别数）
        n_channels: 输入通道数，默认3（RGB）
        bilinear: 是否使用双线性插值上采样，默认False
    """
    return UNet(n_channels, n_classes, bilinear, variant='s')

# 保持向后兼容 - unet 默认使用 small 变体
@register_model('unet')
def create_unet_classifier(n_classes: int, n_channels: int = 3, 
                           bilinear: bool = False, **kwargs):
    """
    UNet分类器（默认使用 small 变体）
    - 输出: [batch_size, n_classes] 的logits
    - 使用全局平均池化将分割图转为分类向量
    
    Args:
        n_classes: 输出类别数
        n_channels: 输入通道数，默认3（RGB）
        bilinear: 是否使用双线性插值上采样，默认False
    """
    class UNetClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet(n_channels, n_classes, bilinear, variant='s')
        
        def forward(self, x):
            seg_map = self.unet(x)
            out = seg_map.mean(dim=(-2, -1))
            return out
    
    return UNetClassifier()

__all__ = ['UNet', 'create_unet_classifier', 'create_unet_segmentation', 'VARIANTS']
