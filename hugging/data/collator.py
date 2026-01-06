"""
Data Collator

用于批处理图像数据
"""

import torch
from PIL import Image
from typing import List, Dict, Any
from .transforms import get_transforms


class ImageCollator:
    """
    图像数据collator
    
    将batch中的图像进行预处理和tensor化
    """
    
    def __init__(self, img_size: int = 224, is_training: bool = True):
        """
        Args:
            img_size: 图像大小
            is_training: 是否为训练模式
        """
        self.img_size = img_size
        self.transforms = get_transforms(img_size, is_training)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            batch: 数据列表，每个元素包含 image_path 和 label_id
            
        Returns:
            包含 pixel_values 和 labels 的字典
        """
        images = []
        labels = []
        
        for item in batch:
            # 加载图像
            img_path = item['image_path']
            img = Image.open(img_path).convert('RGB')
            
            # 应用transforms
            img_tensor = self.transforms(img)
            images.append(img_tensor)
            
            # 获取标签
            labels.append(item['label_id'])
        
        # Stack成batch
        pixel_values = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }
