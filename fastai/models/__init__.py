"""
自定义模型注册模块

使用方法:
    from models import get_model, register_model, list_models
    
    # 获取自定义模型
    model = get_model('unet', n_classes=10)
    
    # 注册新的自定义模型
    @register_model('my_model')
    def create_my_model(n_classes, **kwargs):
        return MyCustomModel(n_classes)

已注册模型:
    - unet: UNet分类模型（全局池化）
    - unet_seg: UNet分割模型（原始输出）
    - yolov11s_cls: YOLOv11-Small 分类模型
    - yolov11m_cls: YOLOv11-Medium 分类模型
    - yolov11l_cls: YOLOv11-Large 分类模型
"""

from .registry import register_model, get_model, list_models, is_custom_model

# 导入所有预定义模型（触发自动注册）
from . import unet
from . import yolo  # ← 新增 YOLO 模型
from . import mlp   # ← 新增 MLP 模型

__all__ = [
    'register_model',
    'get_model',
    'list_models',
    'is_custom_model',
]
