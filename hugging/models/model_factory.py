"""
模型工厂

支持多种架构的模型创建
"""

from transformers import AutoModelForImageClassification, AutoConfig
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from hugging.utils import print_main

class ModelFactory:
    """
    模型工厂类
    
    支持通过架构名称创建模型，兼容多种来源:
    - HuggingFace Hub
    - timm库
    - 自定义模型
    """
    
    # 架构名称到HuggingFace模型的映射
    ARCH_TO_HF_MODEL = {
        'resnet18': 'microsoft/resnet-18',
        'resnet34': 'microsoft/resnet-34',
        'resnet50': 'microsoft/resnet-50',
        'resnet101': 'microsoft/resnet-101',
        'resnet152': 'microsoft/resnet-152',
        'vit_tiny': 'WinKawaks/vit-tiny-patch16-224',
        'vit_small': 'WinKawaks/vit-small-patch16-224',  
        'vit_base': 'google/vit-base-patch16-224',
        'vit_large': 'google/vit-large-patch16-224',
        'swin_tiny': 'microsoft/swin-tiny-patch4-window7-224',
        'swin_small': 'microsoft/swin-small-patch4-window7-224',
        'swin_base': 'microsoft/swin-base-patch4-window7-224',
        'efficientnet_b0': 'google/efficientnet-b0',
        'efficientnet_b1': 'google/efficientnet-b1',
        'efficientnet_b2': 'google/efficientnet-b2',
        'efficientnet_b3': 'google/efficientnet-b3',
        'convnext_tiny': 'facebook/convnext-tiny-224',
        'convnext_small': 'facebook/convnext-small-224',
        'convnext_base': 'facebook/convnext-base-224',
    }
    
    @classmethod
    def create_model(
        cls,
        arch: str,
        num_classes: int,
        pretrained: bool = True
    ) -> nn.Module:
        """
        创建模型
        
        Args:
            arch: 架构名称（如 'resnet18', 'vit_base'）
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
            
        Returns:
            模型实例
        """
        # # 1. 尝试从HuggingFace加载
        # if arch in cls.ARCH_TO_HF_MODEL:
        #     return cls._create_hf_model(arch, num_classes, pretrained)
        
        # 2. 尝试从timm加载
        try:
            return cls._create_timm_model(arch, num_classes, pretrained)
        except Exception as e:
            raise ValueError(
                f"不支持的架构: {arch}\n"
                f"或任何timm支持的模型\n"
                f"错误: {e}"
            )
    
    @classmethod
    def _create_hf_model(
        cls,
        arch: str,
        num_classes: int,
        pretrained: bool
    ) -> nn.Module:
        """从HuggingFace Hub创建模型"""
        hf_model_name = cls.ARCH_TO_HF_MODEL[arch]
        
        print_main(f"📦 从HuggingFace加载: {hf_model_name}")
        
        if pretrained:
            model = AutoModelForImageClassification.from_pretrained(
                hf_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = AutoConfig.from_pretrained(hf_model_name)
            config.num_labels = num_classes
            model = AutoModelForImageClassification.from_config(config)
        
        return model
    
    @classmethod
    def _create_timm_model(
        cls,
        arch: str,
        num_classes: int,
        pretrained: bool
    ) -> nn.Module:
        """从timm库创建模型"""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "需要安装timm库: pip install timm"
            )
        
        print_main(f"📦 从timm加载: {arch}")
        
        # 创建timm模型
        model = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # 包装为HuggingFace兼容的模型
        from .timm_wrapper import TimmModelWrapper
        return TimmModelWrapper(model, num_classes)
    
    @classmethod
    def list_available_models(cls) -> list:
        """列出所有可用的HuggingFace模型"""
        return list(cls.ARCH_TO_HF_MODEL.keys())
