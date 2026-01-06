"""
数据增强和预处理
"""

import torchvision.transforms as T
from typing import Tuple


def get_transforms(
    img_size: int = 224,
    is_training: bool = True
) -> T.Compose:
    """
    获取数据增强transforms
    
    Args:
        img_size: 图像大小
        is_training: 是否为训练模式
        
    Returns:
        transforms组合
    """
    if is_training:
        # 训练时的数据增强
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # 验证时只做基本预处理
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def get_augmentations(img_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """
    获取训练和验证的transforms
    
    Args:
        img_size: 图像大小
        
    Returns:
        (train_transforms, val_transforms)
    """
    train_transforms = get_transforms(img_size, is_training=True)
    val_transforms = get_transforms(img_size, is_training=False)
    
    return train_transforms, val_transforms
