"""YOLOv11 模型注册"""
from ..registry import register_model
from .wrapper import YOLOv11ClassifierWrapper

@register_model('yolov11s_cls')
def create_yolov11s_classifier(n_classes: int, pretrained: bool = True, 
                                model_path: str = None, **kwargs):
    """
    YOLOv11-Small 分类模型
    
    Args:
        n_classes: 输出类别数
        pretrained: 是否使用预训练权重（ImageNet 1000类）
        model_path: 自定义模型路径（如果提供，忽略 pretrained）
    """
    if model_path is None:
        model_path = 'yolo11s-cls.pt' if pretrained else None
    
    return YOLOv11ClassifierWrapper(
        yolo_model_path=model_path,
        n_classes=n_classes
    )

@register_model('yolov11m_cls')
def create_yolov11m_classifier(n_classes: int, pretrained: bool = True,
                                model_path: str = None, **kwargs):
    """
    YOLOv11-Medium 分类模型
    
    Args:
        n_classes: 输出类别数
        pretrained: 是否使用预训练权重（ImageNet 1000类）
        model_path: 自定义模型路径
    """
    if model_path is None:
        model_path = 'yolo11m-cls.pt' if pretrained else None
    
    return YOLOv11ClassifierWrapper(
        yolo_model_path=model_path,
        n_classes=n_classes
    )

@register_model('yolov11l_cls')
def create_yolov11l_classifier(n_classes: int, pretrained: bool = True,
                                model_path: str = None, **kwargs):
    """
    YOLOv11-Large 分类模型
    
    Args:
        n_classes: 输出类别数
        pretrained: 是否使用预训练权重（ImageNet 1000类）
        model_path: 自定义模型路径
    """
    if model_path is None:
        model_path = 'yolo11l-cls.pt' if pretrained else None
    
    return YOLOv11ClassifierWrapper(
        yolo_model_path=model_path,
        n_classes=n_classes
    )

__all__ = [
    'create_yolov11s_classifier',
    'create_yolov11m_classifier',
    'create_yolov11l_classifier',
    'YOLOv11ClassifierWrapper',
]
