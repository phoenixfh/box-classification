from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
import torch
import argparse
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import sys

# 导入自定义模型和工具模块
sys.path.insert(0, str(Path(__file__).parent))
try:
    from models import get_model, is_custom_model
    from utils import dice_score, DiceMetric
    CUSTOM_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  自定义模块导入失败: {e}")
    CUSTOM_MODELS_AVAILABLE = False
    is_custom_model = lambda x: False

def detect_dataset_structure(data_dir):
    """检测数据集的组织结构，支持不同的目录结构模式
    
    支持的结构:
    1. data_dir/train/class1, data_dir/train/class2, ...
    2. data_dir/val/class1, data_dir/val/class2, ...
    3. data_dir/class1/train, data_dir/class2/train, ...
    4. data_dir/class1/val, data_dir/class2/val, ...
    
    Returns:
        tuple: (structure_type, classes)
        structure_type: 1 = 数据集按子集分组, 2 = 数据集按类别分组
        classes: 类别列表
    """
    data_dir = Path(data_dir)
    
    # 检查是否存在train/val子目录，结构类型1
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if val_dir.exists() and val_dir.is_dir():
        # 检查val目录下是否有类别子目录
        class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
        if class_dirs:
            classes = [d.name for d in class_dirs]
            print(f"检测到数据集结构: data_dir/val/class, 类别: {classes}")
            return 1, classes
    
    if train_dir.exists() and train_dir.is_dir():
        # 检查train目录下是否有类别子目录
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if class_dirs:
            classes = [d.name for d in class_dirs]
            print(f"检测到数据集结构: data_dir/train/class, 类别: {classes}")
            return 1, classes
    
    # 检查是否有类别目录，每个类别目录下有train/val，结构类型2
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        train_subdir = subdir / "train"
        val_subdir = subdir / "val"
        if (train_subdir.exists() and train_subdir.is_dir()) or \
           (val_subdir.exists() and val_subdir.is_dir()):
            classes = [d.name for d in subdirs]
            print(f"检测到数据集结构: data_dir/class/train|val, 类别: {classes}")
            return 2, classes
    
    # 尝试第三种结构，检查一级子目录下的二级子目录
    all_subdirs = []
    for subdir in subdirs:
        sub_subdirs = [d for d in subdir.iterdir() if d.is_dir()]
        all_subdirs.extend(sub_subdirs)
    
    # 从所有二级子目录收集潜在的类别
    if all_subdirs:
        potential_classes = set()
        for d in all_subdirs:
            potential_class = d.parent.name
            if potential_class not in ['train', 'val', 'test']:
                potential_classes.add(potential_class)
        
        if potential_classes:
            classes = list(potential_classes)
            print(f"检测到可能的类别: {classes}")
            return 3, classes
    
    # 找不到清晰的结构
    print("无法自动检测数据集结构")
    return 0, []

def build_test_df(root_dir, subset="val"):
    """构建测试数据集的DataFrame"""
    path = Path(root_dir).absolute()
    records = []
    
    # 检测数据集结构
    structure_type, classes = detect_dataset_structure(path)
    
    if structure_type == 1:
        # 数据集按子集分组: data_dir/train/class, data_dir/val/class
        base_path = path / subset
        
        if base_path.exists():
            print(f"使用目录 {base_path} 中的图像进行评估")
            for img_path in base_path.rglob("*.*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                    rel_path = img_path.relative_to(path)
                    # 确保从val目录的直接子目录获取类别名称
                    parts = rel_path.parts
                    if len(parts) >= 2 and parts[0] == subset:
                        class_label = parts[1]  # val/<class_name>/...
                    else:
                        class_label = img_path.parent.name
                    
                    records.append({
                        "filename": str(rel_path),
                        "label": class_label
                    })
        else:
            print(f"警告：{subset}目录不存在，将尝试其他目录结构")
    
    elif structure_type == 2:
        # 数据集按类别分组: data_dir/class/train, data_dir/class/val
        for class_dir in path.iterdir():
            if class_dir.is_dir():
                subset_dir = class_dir / subset
                if subset_dir.exists() and subset_dir.is_dir():
                    for img_path in subset_dir.rglob("*.*"):
                        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                            rel_path = img_path.relative_to(path)
                            class_label = class_dir.name
                            records.append({
                                "filename": str(rel_path),
                                "label": class_label
                            })
    
    # 如果上面的方法都没有找到图像，尝试直接查找所有图像
    if not records:
        print("尝试直接在数据目录中查找图像...")
        val_dir = path / subset
        if val_dir.exists() and val_dir.is_dir():
            # 仅处理val目录中的图像
            for img_path in val_dir.rglob("*.*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                    rel_path = img_path.relative_to(path)
                    parts = rel_path.parts
                    
                    # 确保从val目录的直接子目录获取类别名称
                    if len(parts) >= 2 and parts[0] == subset:
                        class_label = parts[1]  # val/<class_name>/...
                    else:
                        class_label = "unknown"
                        
                    records.append({
                        "filename": str(rel_path),
                        "label": class_label
                    })
        else:
            # 在整个数据目录中查找
            for img_path in path.rglob("*.*"):
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                    rel_path = img_path.relative_to(path)
                    parts = rel_path.parts
                    
                    # 尝试从路径推断类别
                    if subset in parts and len(parts) > parts.index(subset) + 1:
                        # 如果路径包含 'val'，则使用其后的第一个目录作为类别
                        class_label = parts[parts.index(subset) + 1]
                    elif len(parts) >= 2:
                        # 否则使用倒数第二个部分作为类别
                        class_label = parts[-2]
                    else:
                        # 无法确定类别
                        class_label = "unknown"
                    
                    records.append({
                        "filename": str(rel_path),
                        "label": class_label
                    })
    
    print(f"成功创建数据集，包含 {len(records)} 条记录")
    # 输出样本类别信息
    if records:
        sample_labels = list(set([r['label'] for r in records[:min(100, len(records))]])) 
        print(f"数据集包含的类别样本：{sample_labels[:10]}...")
        
    df = pd.DataFrame(records)
    return df, path

def load_model(model_path, arch, device=None, data_path=None, img_size=320):
    """加载模型（支持分类和分割）
    
    Args:
        model_path: 模型路径
        arch: 模型架构名称
        device: 设备类型('cuda'或'cpu')
        data_path: 数据集路径，用于从数据目录获取类别信息
        img_size: 图像尺寸（分割模型需要）
    """
    print("尝试加载模型...")
    
    # 判断是否为分割模型
    is_segmentation = arch.lower().endswith('_seg')
    
    if is_segmentation:
        print(f"🎯 检测到分割模型: {arch}")
        return load_segmentation_model(model_path, arch, device, img_size)
    else:
        print(f"🎯 检测到分类模型: {arch}")
        return load_classification_model(model_path, arch, device, data_path)


def load_segmentation_model(model_path, arch, device=None, img_size=2048):
    """加载分割模型
    
    Args:
        model_path: 模型路径
        arch: 模型架构名称
        device: 设备类型
        img_size: 图像尺寸
    """
    from PIL import Image
    
    # 创建临时数据用于初始化
    print("创建临时分割模型...")
    temp_path = Path('.') / "temp_images"
    temp_path.mkdir(exist_ok=True)
    
    # 创建临时图像
    if not list(temp_path.glob("*.jpg")):
        img = Image.new('RGB', (img_size, img_size), color='white')
        img.save(temp_path / "dummy.jpg")
        mask = Image.new('L', (img_size, img_size), color=0)
        mask.save(temp_path / "dummy_mask.png")
    
    # 使用 get_model 创建分割模型
    if CUSTOM_MODELS_AVAILABLE:
        model = get_model(arch, n_classes=1, n_channels=3)
    else:
        raise ValueError(f"分割模型 '{arch}' 需要自定义模型模块支持")
    
    # 创建简单的数据加载器用于包装
    from torch.utils.data import Dataset, DataLoader
    
    class DummySegDataset(Dataset):
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            img = torch.rand(3, img_size, img_size)
            mask = torch.zeros(img_size, img_size)
            return TensorImage(img), TensorMask(mask)
    
    dummy_ds = DummySegDataset()
    dummy_dl = DataLoader(dummy_ds, batch_size=1)
    
    from fastai.vision.all import DataLoaders
    dls = DataLoaders(dummy_dl, dummy_dl)
    
    # 创建 Learner，需要指定损失函数
    from fastai.learner import Learner
    from torch.nn import BCEWithLogitsLoss
    
    loss_func = BCEWithLogitsLoss()
    learn = Learner(dls, model, loss_func=loss_func)
    
    # 加载权重
    print(f"从 {model_path} 加载权重...")
    state_dict = torch.load(model_path, map_location='cpu' if device != 'cuda' else 'cuda')
    
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    learn.model.load_state_dict(state_dict)
    print("分割模型权重加载成功!")
    
    # 设置为评估模式
    learn.model.eval()
    
    return learn


def load_classification_model(model_path, arch, device=None, data_path=None):
    """加载分类模型
    
    Args:
        model_path: 模型路径
        device: 设备类型('cuda'或'cpu')
        data_path: 数据集路径，用于从数据目录获取类别信息
    """
    print("尝试加载模型...")
    
    # 检查是否有预定义类别
    categories = None
    predefined_classes = os.environ.get('MODEL_CLASSES', None)
    if predefined_classes:
        categories = predefined_classes.split(',')
        print(f"使用预定义类别: {categories}")
    else:
        # 如果提供了数据路径，从数据路径获取类别
        if data_path:
            try:
                # 从指定的数据路径获取类别信息
                print(f"从指定的数据路径获取类别: {data_path}")
                _, categories = detect_dataset_structure(data_path)
                if categories:
                    print(f"从数据路径获取到类别: {categories}")
                else:
                    print("从数据路径未检测到类别，将尝试其他方法...")
            except Exception as dpe:
                print(f"从指定数据路径获取类别失败: {dpe}")
                categories = None
    
    if categories is None:
        print("错误：未指定类别。请使用 --classes 指定类别，或使用 --data_path 指定数据集路径。")
        exit(1)

    # 创建一个临时数据块并构建学习器
    print("创建临时模型...")
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(categories)),
        get_items=get_image_files,
        get_y=lambda x: categories[0]  # 默认类别，稍后会被预测覆盖
    )
    
    # 尝试创建一个临时数据加载器
    path = Path('.')
    temp_path = path / "temp_images"
    temp_path.mkdir(exist_ok=True)
    
    # 确保有至少一个图像进行初始化
    if not list(temp_path.glob("*.jpg")):
        # 创建一个空白图像
        from PIL import Image
        img = Image.new('RGB', (320, 320), color='white')
        img.save(temp_path / "dummy.jpg")

    dls = dblock.dataloaders(temp_path, bs=1)

    print(dls.vocab)
    
    # 使用类别信息创建新的学习器
    # arch = os.environ.get('MODEL_ARCH', 'resnet18')  # 从环境变量获取或使用默认值
    print(f"使用架构 {arch} 创建学习器")
    learn = vision_learner(dls, arch=arch, pretrained=False, n_out=len(dls.vocab))
    
    # 加载权重
    print(f"从 {model_path} 加载权重...")
    state_dict = torch.load(model_path, map_location='cpu' if device != 'cuda' else 'cuda')
    
    print(f"权重键名: {state_dict.keys()}")

    # # 处理可能的DDP状态
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # # 处理可能的带'module.'前缀的权重(DDP)
    # print(f"处理可能的带'module.'前缀的权重(DDP)")
    # if any(k.startswith('module.') for k in state_dict.keys()):
    #     print(k)
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:] if k.startswith('module.') else k
    #         new_state_dict[name] = v
    #     state_dict = new_state_dict
    
    # 加载模型权重
    learn.model.load_state_dict(state_dict)
    print("模型权重加载成功!")
    
    return learn

def predict_segmentation(learn, img_path, img_size=2048):
    """预测分割掩码
    
    Args:
        learn: FastAI Learner
        img_path: 图像路径
        img_size: 图像尺寸
        
    Returns:
        pred_mask: 预测的掩码 (numpy array, 原始图像尺寸)
        original_size: 原始图像尺寸 (width, height)
    """
    from PIL import Image
    
    # 加载并预处理图像
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (width, height)
    img_resized = img.resize((img_size, img_size), Image.BICUBIC)
    
    # 转换为tensor
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # 移到正确的设备
    device = next(learn.model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        pred = learn.model(img_tensor)
        pred_mask_resized = (pred.sigmoid() > 0.5).cpu().numpy()[0, 0]
    
    # 将掩码恢复到原始图像尺寸
    from PIL import Image
    mask_img = Image.fromarray((pred_mask_resized * 255).astype(np.uint8))
    mask_img_original = mask_img.resize(original_size, Image.NEAREST)
    pred_mask = (np.array(mask_img_original) > 127).astype(np.uint8)
    
    return pred_mask, original_size


def evaluate_segmentation_model(learn, data_path, img_size=2048, output_dir=None):
    """评估分割模型
    
    Args:
        learn: FastAI Learner
        data_path: 数据路径
        img_size: 图像尺寸
        output_dir: 输出目录（用于保存可视化结果）
        
    Returns:
        results: 评估结果字典
    """
    from PIL import Image
    
    print(f"开始评估分割模型...")
    data_path = Path(data_path)
    
    # 查找图像和掩码
    img_dir = data_path / 'val' / 'images'
    mask_dir = data_path / 'val' / 'masks'
    
    if not img_dir.exists():
        # 尝试其他结构
        img_dir = data_path / 'images' / 'val'
        mask_dir = data_path / 'masks' / 'val'
    
    if not img_dir.exists():
        raise ValueError(f"找不到图像目录: {img_dir}")
    
    if not mask_dir.exists():
        print(f"⚠️  找不到掩码目录: {mask_dir}，将只进行预测而不评估")
        has_masks = False
    else:
        has_masks = True
    
    # 获取所有图像
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
    
    if len(img_files) == 0:
        raise ValueError(f"在 {img_dir} 中未找到图像")
    
    print(f"找到 {len(img_files)} 张图像")
    
    # 评估
    dice_scores = []
    predictions = []
    
    device = next(learn.model.parameters()).device
    
    for img_path in tqdm(img_files, desc="评估中"):
        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((img_size, img_size), Image.BICUBIC)
            
            # 转换为tensor
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                pred = learn.model(img_tensor)
                pred_prob = pred.sigmoid()
                pred_mask = (pred_prob > 0.5).cpu().numpy()[0, 0]
            
            # 如果有真实掩码，计算Dice score
            dice = None
            if has_masks:
                base_name = img_path.stem
                mask_name = f"{base_name}_mask.png"
                mask_path = mask_dir / mask_name
                
                if not mask_path.exists():
                    mask_name = f"{base_name}.png"
                    mask_path = mask_dir / mask_name
                
                if mask_path.exists():
                    mask = Image.open(mask_path).convert('L')
                    mask_resized = mask.resize((img_size, img_size), Image.NEAREST)
                    mask_array = (np.array(mask_resized) > 127).astype(np.uint8)
                    
                    # 计算Dice score
                    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float().to(device)
                    dice = dice_score(pred, mask_tensor).item()
                    dice_scores.append(dice)
            
            predictions.append({
                'image_path': str(img_path),
                'dice_score': dice,
                'has_mask': dice is not None
            })
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
    
    # 计算统计
    results = {
        'predictions': predictions,
        'num_images': len(img_files),
    }
    
    if dice_scores:
        results['mean_dice'] = float(np.mean(dice_scores))
        results['std_dice'] = float(np.std(dice_scores))
        results['min_dice'] = float(np.min(dice_scores))
        results['max_dice'] = float(np.max(dice_scores))
        
        print(f"\n📊 分割评估结果:")
        print(f"  平均 Dice Score: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
        print(f"  最小 Dice Score: {results['min_dice']:.4f}")
        print(f"  最大 Dice Score: {results['max_dice']:.4f}")
    
    return results


def evaluate_model(learn, test_df, data_path):
    """评估分类模型性能"""
    # 收集真实标签和预测结果
    true_labels = []
    pred_labels = []
    image_paths = []
    probabilities = []
    
    # 直接逐个图像预测
    print("逐图像预测中...")
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="评估中"):
        try:
            # 构建图像路径
            img_path = data_path / row['filename']
            if not img_path.exists():
                print(f"警告：图像文件不存在: {img_path}")
                continue
                
            image_paths.append(str(img_path))
            
            # 获取真实标签
            true_label = row["label"]
            true_labels.append(true_label)
            
            # 预测
            img = PILImage.create(img_path)
            pred_class, pred_idx, probs = learn.predict(img)
#            print(pred_class, pred_idx, probs )
            pred_labels.append(str(pred_class))
            probabilities.append({str(c): float(p) for c, p in zip(learn.dls.vocab, map(float, probs))})
        except Exception as e:
            print(f"预测图像 {row['filename']} 时出错: {e}")
    
    # 计算性能指标
    if not true_labels or not pred_labels:
        raise ValueError("没有有效的预测结果")
        
    # 对齐标签
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    # 使用共同标签计算性能指标
    report = classification_report(true_labels, pred_labels, labels=unique_labels, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # 组织结果
    results = {
        "individual_predictions": [
            {
                "image_path": path,
                "true_label": true,
                "predicted_label": pred,
                "probabilities": prob
            }
            for path, true, pred, prob in zip(image_paths, true_labels, pred_labels, probabilities)
        ],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": unique_labels
    }
    
    return results

def save_results(results, output_path, is_segmentation=False):
    """保存预测结果和评估指标
    
    Args:
        results: 评估结果字典
        output_path: 输出路径
        is_segmentation: 是否为分割任务
    """
    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存JSON结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_path}")
    
    # 如果是分割任务，不需要生成分类报告图表
    if is_segmentation:
        print("分割任务评估完成")
        return
    
    # 创建报告目录
    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # 提取分类报告
    report = results["classification_report"]
    
    # 生成性能指标图表
    metrics = ['precision', 'recall', 'f1-score']
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        # 排除'accuracy', 'macro avg', 'weighted avg'
        metric_data = {k: report[k][metric] for k in report if k not in ['accuracy', 'macro avg', 'weighted avg']}
        
        # 按值排序显示
        sorted_data = {k: v for k, v in sorted(metric_data.items(), key=lambda item: item[1], reverse=True)}
        
        # 创建条形图
        plt.bar(sorted_data.keys(), sorted_data.values())
        plt.title(f"{metric.capitalize()} by Class")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()
    
    plt.savefig(report_dir / "metrics_by_class.png")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(16, 14))
    labels = results["confusion_matrix_labels"]
    cm = np.array(results["confusion_matrix"])
    
    # 计算归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # 处理除以0的情况
    
    ax = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png")
    
    print(f"报告图表已保存到 {report_dir}")
    
    # 打印总体性能
    print("\n模型总体性能:")
    print(f"准确率(Accuracy): {report['accuracy']:.4f}")
    print(f"宏平均(Macro Avg) - 精确率: {report['macro avg']['precision']:.4f}, 召回率: {report['macro avg']['recall']:.4f}, F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"加权平均(Weighted Avg) - 精确率: {report['weighted avg']['precision']:.4f}, 召回率: {report['weighted avg']['recall']:.4f}, F1-Score: {report['weighted avg']['f1-score']:.4f}")

def predict_single_image(model_path, image_path, arch='resnet18', device=None, classes=None, data_path=None, img_size=320, save_mask=False):
    """预测单张图片的类别或分割掩码
    
    Args:
        model_path: 模型路径(.pth 或 .pkl)
        image_path: 图片路径
        arch: 模型架构
        device: 运行设备 ('cuda' 或 'cpu')
        classes: 类别列表，如果为None则尝试从模型中获取
        data_path: 数据集路径，用于从数据目录获取类别信息
        img_size: 图像尺寸
        save_mask: 是否保存预测掩码（分割任务）
    """
    print(f"\n===== 预测单张图片 =====")
    print(f"图片: {image_path}")
    print(f"模型: {model_path}")
    print(f"架构: {arch}")
    if data_path:
        print(f"数据路径: {data_path}")
    
    # 确保图片存在
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"错误: 找不到图片: {image_path}")
        return
    
    # 判断是否为分割模型
    is_segmentation = arch.lower().endswith('_seg')
    
    # 加载模型
    start_time = time.time()
    
    try:
        # 加载模型
        learn = load_model(model_path, arch, device, data_path, img_size)
        load_time = time.time() - start_time
        
        if is_segmentation:
            # 分割任务预测
            inference_start = time.time()
            pred_mask, original_size = predict_segmentation(learn, img_path, img_size)
            inference_time = time.time() - inference_start
            
            print(f"\n预测结果:")
            print(f"原始图像尺寸: {original_size}")
            print(f"掩码形状: {pred_mask.shape}")
            print(f"前景像素占比: {pred_mask.sum() / pred_mask.size * 100:.2f}%")
            
            # 保存预测掩码
            if save_mask:
                from PIL import Image
                output_path = Path.cwd() / f"{img_path.stem}_pred_mask.png"
                mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                mask_img.save(output_path)
                print(f"预测掩码已保存到: {output_path}")
            
        else:
            # 分类任务预测
            img = PILImage.create(img_path)
            
            inference_start = time.time()
            pred_class, pred_idx, probs = learn.predict(img)
            inference_time = time.time() - inference_start
            
            print(f"输出: {pred_class} {pred_idx} {probs}")
            print(f"\n预测结果:")
            print(f"类别: {pred_class}")
            print(f"置信度: {float(probs[pred_idx]):.4f}")
            
            # 显示前5个最高概率的类别
            top_k_values, top_k_indices = torch.topk(probs, min(5, len(probs)))
            
            print("\n前5个预测:")
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_values), 1):
                print(f"{i}. {learn.dls.vocab[idx]}: {float(prob):.4f}")
        
        print(f"\n模型加载时间: {load_time*1000:.2f}ms")
        print(f"推理时间: {inference_time*1000:.2f}ms")
        print(f"总时间: {(time.time() - start_time)*1000:.2f}ms")
        
    except Exception as e:
        print(f"预测时出错: {e}")
        import traceback
        traceback.print_exc()


def load_onnx_model(model_path, device=None, data_path=None):
    """加载ONNX模型并确保类别顺序一致"""
    print(f"尝试加载ONNX模型: {model_path}")
    
    try:
        import onnx
        
        # 设置执行提供程序
        if device == 'cuda' and 'CUDAExecutionProvider' in onnx.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("使用CUDA执行ONNX模型")
        else:
            providers = ['CPUExecutionProvider']
            print("使用CPU执行ONNX模型")
        
        # 加载ONNX模型
        model = onnx.load(model_path)

        # 获取类别信息的优先级：
        # 1. 从模型元数据中获取
        # 2. 从环境变量获取
        # 3. 从数据路径获取
        
        categories = None
        class_indices_map = None
        
        # 1. 首先尝试从模型元数据获取
        for meta in model.metadata_props:
            if meta.key == "classes":
                categories = meta.value.split(",")
                print(f"从ONNX模型元数据获取类别: {categories}")
                break
                
            # 检查是否有带索引的类别信息
            if meta.key == "class_indices" and not categories:
                index_class_pairs = meta.value.split(",")
                class_indices_map = {}
                for pair in index_class_pairs:
                    idx, class_name = pair.split(":")
                    class_indices_map[int(idx)] = class_name
                
                # 按索引排序类别
                if class_indices_map:
                    categories = [class_indices_map[i] for i in range(len(class_indices_map))]
                    print(f"从ONNX模型元数据获取有序类别: {categories}")
        
        # 2. 尝试从环境变量获取
        if not categories:
            predefined_classes = os.environ.get('MODEL_CLASSES', None)
            if predefined_classes:
                categories = predefined_classes.split(',')
                print(f"使用预定义类别: {categories}")
        
        # 3. 从数据路径获取
        if not categories and data_path:
            try:
                print(f"从数据路径获取类别: {data_path}")
                # 使用与训练时相同的方法构建类别列表
                test_df, _ = build_test_df(data_path)
                if len(test_df) > 0:
                    # 确保按字母排序，与大多数数据集加载器一致
                    categories = sorted(test_df['label'].unique().tolist())
                    print(f"从数据路径获取并排序类别: {categories}")
            except Exception as dpe:
                print(f"从数据路径获取类别失败: {dpe}")
        
        # 如果仍然没有类别，给出错误
        if not categories:
            print("错误: 未找到类别信息。请使用--classes参数指定类别，或使用--data_path指定数据集路径。")
            exit(1)

        # 使用onnxruntime运行模型
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=providers)

        input_name = session.get_inputs()[0].name

        return session, input_name, categories
        
    except ImportError:
        print("错误: 未安装onnx。请使用pip install onnx安装。")
        exit(1)
    except Exception as e:
        print(f"加载ONNX模型出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

def preprocess_image_for_onnx(img_path, img_size=320):
    """预处理图像用于ONNX模型推理
    
    Args:
        img_path: 图像路径
        img_size: 输入图像大小
    
    Returns:
        preprocessed_img: 预处理后的图像（numpy数组）
    """
    from PIL import Image
    import numpy as np
    
    # 打开图像
    img = Image.open(img_path).convert('RGB')
    
    # 调整大小
    img = img.resize((img_size, img_size), Image.LANCZOS)
    
    # 转换为numpy数组，并进行标准化 [0,1]
    img_array = np.array(img, dtype=np.float32) / 255.0  # 显式指定为float32
    
    # 转换为NCHW格式 (批次, 通道, 高度, 宽度)
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 最后确认一次数据类型
    img_array = img_array.astype(np.float32)
    
    # 打印数组信息以便调试
    print(f"预处理后图像形状: {img_array.shape}, 数据类型: {img_array.dtype}")
    
    return img_array

def predict_with_onnx(session, input_name, img_array, categories):
    """使用ONNX模型进行预测
    
    Args:
        session: ONNX运行时会话
        input_name: 模型输入名称
        img_array: 预处理后的图像数组
        categories: 类别列表
    
    Returns:
        pred_class: 预测的类别
        pred_idx: 预测的类别索引
        probs: 所有类别的概率
    """
    import numpy as np
    
    # 确保输入是float32类型
    if img_array.dtype != np.float32:
        print(f"警告: 输入数组类型为 {img_array.dtype}，转换为 float32")
        img_array = img_array.astype(np.float32)
    
    # 准备输入
    input_dict = {input_name: img_array}
    
    # # 获取模型预期的输入类型
    # input_details = session.get_inputs()
    # print(f"模型输入详情: 名称={input_details[0].name}, 类型={input_details[0].type}, 形状={input_details[0].shape}")
    
    # 执行推理
    try:
        outputs = session.run(None, input_dict)
    except Exception as e:
        print(f"ONNX推理出错: {e}")
        print(f"输入数组信息: 形状={img_array.shape}, 类型={img_array.dtype}, 数值范围=[{np.min(img_array)}, {np.max(img_array)}]")
        raise
    
    # 获取输出（通常是logits）
    logits = outputs[0]
    # print(f"输出logits形状: {logits.shape}, 类型: {logits.dtype}, {outputs}")
    
    # 应用softmax获取概率
    # 使用更稳定的softmax实现
    logits = logits - np.max(logits, axis=1, keepdims=True)  # 为了数值稳定性
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs = probs[0]  # 获取第一个批次的结果
    
    # 获取预测类别
    pred_idx = np.argmax(probs)
    pred_class = categories[pred_idx]
    
    return pred_class, pred_idx, probs

def predict_single_image_onnx(model_path, image_path, device=None, img_size=320, classes=None, data_path=None):
    """使用ONNX模型预测单张图片
    
    Args:
        model_path: ONNX模型路径
        image_path: 图片路径
        device: 运行设备 ('cuda' 或 'cpu')
        img_size: 图像大小
        classes: 类别列表，如果为None则尝试从模型中获取
        data_path: 数据集路径，用于从数据目录获取类别信息
    """
    print(f"\n===== 使用ONNX模型预测单张图片 =====")
    print(f"图片: {image_path}")
    print(f"模型: {model_path}")
    if data_path:
        print(f"数据路径: {data_path}")
    
    # 确保图片存在
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"错误: 找不到图片: {image_path}")
        return
    
    # 如果提供了类别列表，使用它
    if classes:
        categories = [c.strip() for c in classes.split(',')]
        os.environ['MODEL_CLASSES'] = ','.join(categories)
        print(f"使用指定的类别列表: {categories}")
    
    # 加载模型
    start_time = time.time()
    
    try:
        # 加载ONNX模型
        session, input_name, categories = load_onnx_model(model_path, device, data_path)
        load_time = time.time() - start_time
        
        # 预处理图像
        preprocess_start = time.time()
        img_array = preprocess_image_for_onnx(img_path, img_size)
        preprocess_time = time.time() - preprocess_start
        
        # 执行推理
        inference_start = time.time()
        pred_class, pred_idx, probs = predict_with_onnx(session, input_name, img_array, categories)
        inference_time = time.time() - inference_start
        
        # 输出结果
        print(f"\n预测结果:")
        print(f"类别: {pred_class}")
        print(f"置信度: {float(probs[pred_idx]):.4f}")
        
        # 显示前5个最高概率的类别
        top_k_indices = np.argsort(probs)[::-1]
        
        print("\n预测:")
        for i, idx in enumerate(top_k_indices, 1):
            print(f"{i}. {categories[idx]}: {float(probs[idx]):.4f}")
        
        print(f"\n模型加载时间: {load_time*1000:.2f}ms")
        print(f"图像预处理时间: {preprocess_time*1000:.2f}ms")
        print(f"推理时间: {inference_time*1000:.2f}ms")
        print(f"总时间: {(time.time() - start_time)*1000:.2f}ms")
        
    except Exception as e:
        print(f"预测时出错: {e}")
        import traceback
        traceback.print_exc()

def predict_single_image_onnx_seg(model_path, image_path, device=None, img_size=2048, save_mask=True):
    """使用ONNX模型预测单张图片（分割任务）
    
    Args:
        model_path: ONNX模型路径
        image_path: 图片路径
        device: 运行设备 ('cuda' 或 'cpu')
        img_size: 图像大小
        save_mask: 是否保存预测掩码
    """
    print(f"\n===== 使用ONNX模型预测单张图片（分割）=====")
    print(f"图片: {image_path}")
    print(f"模型: {model_path}")
    
    # 确保图片存在
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"错误: 找不到图片: {image_path}")
        return
    
    # 加载模型
    start_time = time.time()
    
    try:
        import onnxruntime as ort
        from PIL import Image
        
        # 设置执行提供程序
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("使用CUDA执行ONNX模型")
        else:
            providers = ['CPUExecutionProvider']
            print("使用CPU执行ONNX模型")
        
        # 加载ONNX模型
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        load_time = time.time() - start_time
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        original_size = img.size  # (width, height)
        
        # 预处理图像
        preprocess_start = time.time()
        img_resized = img.resize((img_size, img_size), Image.BICUBIC)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        preprocess_time = time.time() - preprocess_start
        
        # 执行推理
        inference_start = time.time()
        input_dict = {input_name: img_array}
        outputs = session.run(None, input_dict)
        inference_time = time.time() - inference_start
        
        # 获取输出
        logits = outputs[0][0, 0]  # [H, W]
        
        # 应用sigmoid并二值化
        pred_mask_resized = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.uint8)
        
        # 恢复到原始尺寸
        mask_img = Image.fromarray((pred_mask_resized * 255).astype(np.uint8))
        mask_img_original = mask_img.resize(original_size, Image.NEAREST)
        pred_mask = (np.array(mask_img_original) > 127).astype(np.uint8)
        
        # 输出结果
        print(f"\n预测结果:")
        print(f"原始图像尺寸: {original_size}")
        print(f"掩码形状: {pred_mask.shape}")
        print(f"前景像素占比: {pred_mask.sum() / pred_mask.size * 100:.2f}%")
        
        # 保存预测掩码
        if save_mask:
            output_path = Path.cwd() / f"{img_path.stem}_pred_mask.png"
            mask_img_save = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_img_save.save(output_path)
            print(f"预测掩码已保存到: {output_path}")
        
        print(f"\n模型加载时间: {load_time*1000:.2f}ms")
        print(f"图像预处理时间: {preprocess_time*1000:.2f}ms")
        print(f"推理时间: {inference_time*1000:.2f}ms")
        print(f"总时间: {(time.time() - start_time)*1000:.2f}ms")
        
    except ImportError:
        print("错误: 未安装onnxruntime。请使用pip install onnxruntime安装。")
    except Exception as e:
        print(f"预测时出错: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model_onnx(model_path, test_df, data_path, device=None, img_size=320, classes=None):
    """使用ONNX模型评估数据集
    
    Args:
        model_path: ONNX模型路径
        test_df: 测试数据集的DataFrame
        data_path: 数据集路径
        device: 运行设备 ('cuda' 或 'cpu')
        img_size: 图像大小
        classes: 类别列表，如果为None则尝试从模型中获取
    
    Returns:
        results: 评估结果字典
    """
    
    # 如果提供了类别列表，使用它
    if classes:
        categories = [c.strip() for c in classes.split(',')]
        os.environ['MODEL_CLASSES'] = ','.join(categories)
    
    # 加载ONNX模型
    session, input_name, categories = load_onnx_model(model_path, device, data_path)
    
    # 收集真实标签和预测结果
    true_labels = []
    pred_labels = []
    image_paths = []
    probabilities = []
    
    # 直接逐个图像预测
    print("逐图像预测中...")
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="评估中"):
        try:
            # 构建图像路径
            img_path = data_path / row['filename']
            if not img_path.exists():
                print(f"警告：图像文件不存在: {img_path}")
                continue
                
            image_paths.append(str(img_path))
            
            # 获取真实标签
            true_label = row["label"]
            true_labels.append(true_label)
            
            # 预处理图像
            img_array = preprocess_image_for_onnx(img_path, img_size)
            
            # 执行推理
            pred_class, pred_idx, probs = predict_with_onnx(session, input_name, img_array, categories)
            
            pred_labels.append(str(pred_class))
            probabilities.append({str(c): float(p) for c, p in zip(categories, map(float, probs))})
        except Exception as e:
            print(f"预测图像 {row['filename']} 时出错: {e}")
    
    # 计算性能指标
    if not true_labels or not pred_labels:
        raise ValueError("没有有效的预测结果")
        
    # 对齐标签
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    # 使用共同标签计算性能指标
    report = classification_report(true_labels, pred_labels, labels=unique_labels, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # 组织结果
    results = {
        "individual_predictions": [
            {
                "image_path": path,
                "true_label": true,
                "predicted_label": pred,
                "probabilities": prob
            }
            for path, true, pred, prob in zip(image_paths, true_labels, pred_labels, probabilities)
        ],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": unique_labels
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='模型预测与评估')
    
    # 基本参数
    parser.add_argument('--data_path', type=str, help='数据集路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--output_path', type=str, default='results/evaluation_results.json', help='输出结果路径')
    parser.add_argument('--subset', type=str, default='val', help='测试子集目录(默认为 val)')
    
    # 图像处理参数
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    
    # 模型参数
    parser.add_argument('--arch', type=str, default='resnet18', help='模型架构，用于重建模型（如resnet18、resnet34等）')
    parser.add_argument('--classes', type=str, help='类别列表，用逗号分隔（例如"cat,dog,horse"）')
    
    # 设备参数
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None, 
                       help='运行设备。默认自动选择，如果有GPU则使用GPU')

    # 操作模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--image', type=str, help='要预测的单张图片路径')
    
    # 分割任务相关
    parser.add_argument('--save_mask', action='store_true', help='保存预测的分割掩码')
    
    args = parser.parse_args()
    
    # 判断是否为分割模型
    is_segmentation = args.arch.lower().endswith('_seg')
    
    # 如果是分割任务且未指定img_size，使用更大的默认值
    if is_segmentation and args.img_size == 320:
        args.img_size = 2048
        print(f"分割任务，使用默认图像尺寸: {args.img_size}")
    
    # # 设置环境变量，用于模型重建
    # os.environ['MODEL_ARCH'] = args.arch
    
    # 如果提供了类别列表，解析并存储在环境变量中
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
        os.environ['MODEL_CLASSES'] = ','.join(classes)
        print(f"使用指定的类别列表: {classes}")
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告：CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    # 判断模型类型
    is_onnx_model = 'onnx' in args.model_path.lower()
    
    # 单张图片预测
    if args.image:
        if is_onnx_model:
            if is_segmentation:
                predict_single_image_onnx_seg(
                    model_path=args.model_path,
                    image_path=args.image,
                    device=args.device,
                    img_size=args.img_size
                )
            else:
                predict_single_image_onnx(
                    model_path=args.model_path,
                    image_path=args.image,
                    device=args.device,
                    img_size=args.img_size,
                    classes=args.classes,
                    data_path=args.data_path
                )
        else:
            predict_single_image(
                model_path=args.model_path,
                image_path=args.image,
                arch=args.arch,
                device=args.device,
                img_size=args.img_size,
                classes=args.classes,
                data_path=args.data_path,
                save_mask=args.save_mask
            )
        return
    
    # 如果没有指定操作模式，执行数据集评估
    if not args.data_path:
        print("错误：未指定数据集路径。请使用 --data_path 指定数据集路径，或使用 --image 进行单张图片预测。")
        return
    
    # 分割任务的数据集评估
    if is_segmentation:
        print("开始评估分割模型...")
        try:
            learn = load_model(args.model_path, args.arch, args.device, args.data_path, args.img_size)
            print(f"分割模型加载成功")
            
            # 评估分割模型
            output_dir = Path(args.output_path).parent
            results = evaluate_segmentation_model(learn, args.data_path, args.img_size, output_dir)
            
            # 保存结果
            save_results(results, args.output_path, is_segmentation=True)
        except Exception as e:
            print(f"分割模型评估失败：{e}")
            import traceback
            traceback.print_exc()
            return
    
    else:
        # 分类任务的数据集评估
        # 构建测试数据集
        test_df, data_path = build_test_df(args.data_path, args.subset)
        print(f"找到 {len(test_df)} 张测试图片")
        
        if len(test_df) == 0:
            print(f"错误：在 {args.data_path}/{args.subset} 目录下未找到图片")
            return
        
        # 如果没有指定类别，从测试数据集中获取
        if not args.classes and 'MODEL_CLASSES' not in os.environ:
            detected_classes = test_df['label'].unique().tolist()
            if len(detected_classes) > 0:
                os.environ['MODEL_CLASSES'] = ','.join(detected_classes)
                print(f"从测试数据集检测到的类别: {detected_classes}")
        
        # 根据模型类型进行评估
        if is_onnx_model:
            # 使用ONNX模型评估
            print("使用ONNX模型进行评估...")
            results = evaluate_model_onnx(
                model_path=args.model_path, 
                test_df=test_df, 
                data_path=data_path, 
                device=args.device, 
                img_size=args.img_size, 
                classes=args.classes
            )
        else:
            # 使用PyTorch模型评估
            try:
                learn = load_model(args.model_path, args.arch, args.device, args.data_path, args.img_size)
                print(f"模型加载成功，类别：{learn.dls.vocab}")
                
                # 评估模型
                print("开始评估模型...")
                results = evaluate_model(learn, test_df, data_path)
            except Exception as e:
                print(f"无法加载模型：{e}")
                import traceback
                traceback.print_exc()
                return
        
        # 保存结果
        save_results(results, args.output_path, is_segmentation=False)

if __name__ == '__main__':
    main() 
