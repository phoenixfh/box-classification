"""
图像数据集加载器（参考 fastai/train.py 的简化实现）

支持标准的train/val目录结构，使用 pickle 缓存
"""

import os
from datasets import Dataset
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any
from hugging.utils import print_main


class ImageDataset:
    """
    图像数据集加载器（参考 fastai/train.py）
    
    支持的目录结构:
    data_path/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── val/
        ├── class1/
        ├── class2/
        └── ...
    """
    
    @staticmethod
    def from_directory(
        data_path: str,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        img_size: int = 224,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        从目录结构加载数据集（参考 fastai/train.py 的实现）
        
        每个进程独立加载，使用简单的 pickle 缓存
        
        Args:
            data_path: 数据集根目录
            train_size: 训练集大小限制（可选）
            val_size: 验证集大小限制（可选）
            img_size: 图像大小
            use_cache: 是否使用缓存（默认True）
            
        Returns:
            包含train、val数据集和元信息的字典
        """
        path = Path(data_path).absolute()
        cache_dir = path / '.dataset_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        def build_df(subset: str, size_limit: Optional[int] = None) -> pd.DataFrame:
            """构建数据集DataFrame，使用 pickle 缓存"""
            # 缓存文件路径
            cache_key = f"{subset}_{img_size}"
            if size_limit:
                cache_key += f"_{size_limit}"
            cache_file = cache_dir / f"{cache_key}.pkl"
            
            # 如果缓存存在，直接加载
            if use_cache and cache_file.exists():
                print_main(f"✓ 使用缓存的 {subset} 数据集")
                return pd.read_pickle(cache_file)
            
            # 构建数据集
            print_main(f"⚙️  构建 {subset} 数据集...")
            
            records = []
            subset_path = path / subset
            
            if not subset_path.exists():
                raise ValueError(f"目录不存在: {subset_path}")
            
            for class_dir in subset_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                images = (
                    list(class_dir.glob('*.jpg')) + 
                    list(class_dir.glob('*.png')) +
                    list(class_dir.glob('*.jpeg'))
                )
                
                for img_path in images:
                    records.append({
                        'image_path': str(img_path),
                        'label': class_name
                    })
            
            df = pd.DataFrame(records)
            
            # 限制大小
            if size_limit and len(df) > size_limit:
                df = df.sample(n=size_limit, random_state=42)
            
            # 保存缓存
            if use_cache:
                df.to_pickle(cache_file)
                print_main(f"✓ {subset} 数据集构建完成: {len(df)} 张图片（已缓存）")
            else:
                print_main(f"✓ {subset} 数据集构建完成: {len(df)} 张图片")
            
            return df
        
        # 加载数据
        train_df = build_df('train', train_size)
        val_df = build_df('val', val_size)
        
        # 🔧 关键修复: 打乱验证集
        print_main("🔀 打乱验证集以确保分布式训练时的准确性...")
        val_df = val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print_main(f"  训练集: {len(train_df)} 样本")
        print_main(f"  验证集: {len(val_df)} 样本")
        
        # 创建标签映射
        all_labels = sorted(set(train_df['label'].unique()) | set(val_df['label'].unique()))
        label2id = {label: i for i, label in enumerate(all_labels)}
        id2label = {i: label for label, i in label2id.items()}
        
        print_main(f"  类别数: {len(all_labels)}")
        
        # 转换为HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # 添加label_id列
        def add_label_id(example):
            example['label_id'] = label2id[example['label']]
            return example
        
        train_dataset = train_dataset.map(add_label_id)
        val_dataset = val_dataset.map(add_label_id)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'num_classes': len(all_labels),
            'label2id': label2id,
            'id2label': id2label,
            'labels': all_labels
        }
