"""YOLOv11 分类模型包装器 - 适配 FastAI"""
import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv11ClassifierWrapper(nn.Module):
    """
    YOLOv11 分类模型包装器
    
    将 YOLO 的 tuple 输出适配为 FastAI 期望的 tensor 输出
    """
    def __init__(self, yolo_model_path: str = None, n_classes: int = None):
        """
        Args:
            yolo_model_path: YOLO 模型路径（.pt 文件）
            n_classes: 输出类别数（如果与模型不同，会替换分类头）
        """
        super().__init__()
        
        # 加载 YOLO 模型
        if yolo_model_path is None:
            yolo_model_path = 'yolo11s-cls.pt'  # 默认使用 small 模型

        yolo = YOLO(yolo_model_path)
        self.model = yolo.model  # 获取 PyTorch 模型
        self.original_n_classes = len(yolo.names) if hasattr(yolo, 'names') else 1000

        if n_classes is not None and n_classes != self.original_n_classes:
            self._replace_classifier(n_classes)
            self.n_classes = n_classes
        else:
            self.n_classes = self.original_n_classes

    def _replace_classifier(self, n_classes: int):
        """替换 YOLO 的分类头以支持自定义类别数"""
        # YOLO-cls 的分类头在 model.model[-1].linear
        if hasattr(self.model.model[-1], 'linear'):
            in_features = self.model.model[-1].linear.in_features
            self.model.model[-1].linear = nn.Linear(in_features, n_classes)
            print(f"✅ 替换分类头: {self.original_n_classes} 类 → {n_classes} 类")
        else:
            print(f"⚠️  无法找到分类头，保持原始类别数: {self.original_n_classes}")
    
    def forward(self, x):
        """
        前向传播，返回标准的 logits tensor
        
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        
        Returns:
            logits: [batch_size, n_classes]
        """
        output = self.model(x)
        
        # YOLO 返回 tuple: (logits, softmax_output)
        # 我们只取 logits
        if isinstance(output, tuple):
            return output[0]  # [batch_size, n_classes]
        else:
            return output
