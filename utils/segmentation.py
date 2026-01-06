"""
图像分割任务专用数据加载模块
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from fastai.vision.all import TensorImage, TensorMask, TfmdDL, DataLoaders

from .data_loading import is_main_process


class SegmentationDataset(TorchDataset):
    """UNet 分割数据集（实时加载 + 可选磁盘缓存）"""
    def __init__(self, img_dir, mask_dir, img_files, img_size=2048, scale=1.0, use_disk_cache=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_files = img_files
        self.img_size = img_size
        self.scale = scale
        self.use_disk_cache = use_disk_cache
        
        # 磁盘缓存目录（NPZ 压缩格式）
        if use_disk_cache:
            cache_root = img_dir.parent.parent / '.cache'
            self.disk_cache_dir = cache_root / f'npz_{img_dir.parent.name}_{img_dir.name}_size{img_size}_scale{scale}'
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_cache_dir = None
        
    def __len__(self):
        return len(self.img_files)
    
    def _load_and_preprocess(self, img_name):
        """加载并预处理单个样本"""
        # 加载图像
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert('RGB')
        
        # 加载mask
        base_name = img_name.rsplit('.', 1)[0]
        mask_name = f"{base_name}_mask.png"
        mask_path = self.mask_dir / mask_name
        
        if not mask_path.exists():
            mask_name = f"{base_name}.png"
            mask_path = self.mask_dir / mask_name
        
        mask = Image.open(mask_path).convert('L')
        
        # 预处理：缩放
        if self.scale != 1.0:
            w, h = img.size
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # Resize到目标大小
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # 转换为numpy
        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 127).astype(np.uint8)
        
        return img_np, mask_np
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        
        # 尝试从磁盘缓存加载
        if self.use_disk_cache and self.disk_cache_dir:
            cache_file = self.disk_cache_dir / f'{idx:05d}.npz'
            
            if cache_file.exists():
                try:
                    data = np.load(cache_file)
                    img_np = data['img']
                    mask_np = data['mask']
                except Exception:
                    # 缓存损坏，重新加载
                    img_np, mask_np = self._load_and_preprocess(img_name)
                    np.savez_compressed(cache_file, img=img_np, mask=mask_np)
            else:
                # 缓存不存在，实时加载并保存
                img_np, mask_np = self._load_and_preprocess(img_name)
                try:
                    np.savez_compressed(cache_file, img=img_np, mask=mask_np)
                except Exception:
                    pass  # 保存失败不影响训练
        else:
            # 不使用缓存，直接实时加载
            img_np, mask_np = self._load_and_preprocess(img_name)
        
        # 转换为tensor
        img = TensorImage(torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0)
        mask = TensorMask(torch.from_numpy(mask_np.astype(np.int64)))
        
        return img, mask


def get_segmentation_dls(data_dir, batch_size=4, img_size=2048, scale=0.5, num_workers=8, use_disk_cache=True):
    """
    创建分割任务的 DataLoaders（磁盘缓存）
    
    Args:
        data_dir: 数据根目录，应包含 imgs/ 和 masks/ 子目录
        batch_size: 批次大小
        img_size: 图像尺寸
        scale: 图像缩放比例（< 1.0 节省显存）
        num_workers: DataLoader 工作进程数
        use_disk_cache: 是否使用磁盘缓存
        
    Returns:
        DataLoaders: FastAI DataLoaders 对象
    """
    data_dir = Path(data_dir)
    
    # 获取训练和验证图像列表
    train_img_dir = data_dir / 'imgs' / 'train'
    val_img_dir = data_dir / 'imgs' / 'val'
    train_mask_dir = data_dir / 'masks' / 'train'
    val_mask_dir = data_dir / 'masks' / 'val'
    
    train_imgs = sorted([f.name for f in train_img_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
    val_imgs = sorted([f.name for f in val_img_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
    
    if is_main_process():
        print(f"  训练集图像数量: {len(train_imgs)}")
        print(f"  验证集图像数量: {len(val_imgs)}")
        if use_disk_cache:
            print(f"  💾 磁盘缓存: 启用（NPZ压缩格式，按需创建）")
        else:
            print(f"  ⚡ 实时加载: 启用（无缓存）")
    
    # 创建数据集（启用磁盘缓存）
    train_ds = SegmentationDataset(train_img_dir, train_mask_dir, train_imgs, img_size, scale, use_disk_cache)
    val_ds = SegmentationDataset(val_img_dir, val_mask_dir, val_imgs, img_size, scale, use_disk_cache)
    
    # 创建 DataLoaders
    train_dl = TfmdDL(train_ds, batch_size=batch_size, shuffle=True, 
                      num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = TfmdDL(val_ds, batch_size=batch_size, shuffle=False, 
                    num_workers=num_workers, pin_memory=True, drop_last=True)
    
    dls = DataLoaders(train_dl, val_dl)
    return dls
