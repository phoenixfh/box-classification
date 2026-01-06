#!/usr/bin/env python
"""
独立的模型评估脚本
支持对训练好的模型进行评估，生成混淆矩阵和分类报告

用法:
    # 基本用法
    python fastai/evaluate.py --model best.pth --data /path/to/data
    
    # 完整参数（关联到训练 Run）
    python fastai/evaluate.py \
        --model runs/ai-classifier/resnet18/best.pth \
        --data /mnt/ssd/dataset \
        --img_size 224 \
        --batch_size 256 \
        --arch resnet18 \
        --output_dir ./evaluation_results \
        --mlflow_run_id <run_id>
"""

from fastai.vision.all import *
from pathlib import Path
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# 导入公共工具函数
sys.path.insert(0, str(Path(__file__).parent))
from utils import upload_figure_to_mlflow, upload_metrics_to_mlflow, upload_artifact_to_mlflow

def load_model_from_checkpoint(checkpoint_path, data_path, img_size, batch_size, arch):
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: 模型checkpoint路径
        data_path: 数据集路径
        img_size: 图像尺寸
        batch_size: batch大小
        arch: 模型架构
        
    Returns:
        learn: 加载好权重的Learner对象
    """
    # 关键修复：清理分布式相关的环境变量
    # 评估作为独立进程运行，不应继承训练进程的分布式设置
    import os
    distributed_env_vars = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 
        'MASTER_ADDR', 'MASTER_PORT',
        'TORCH_DISTRIBUTED_DEBUG'
    ]
    
    cleaned_vars = []
    for var in distributed_env_vars:
        if var in os.environ:
            del os.environ[var]
            cleaned_vars.append(var)
    
    if cleaned_vars:
        print(f"🔧 清理分布式环境变量: {', '.join(cleaned_vars)}")
    
    print(f"\n{'='*80}")
    print("加载模型")
    print(f"{'='*80}")
    
    checkpoint_path = Path(checkpoint_path)
    data_path = Path(data_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")
    
    # 1. 加载checkpoint
    print(f"📦 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取checkpoint中的信息
    saved_img_size = checkpoint.get('img_size', img_size)
    saved_arch = checkpoint.get('arch', arch)
    
    print(f"   Checkpoint信息:")
    print(f"   - 架构: {saved_arch}")
    print(f"   - 图像尺寸: {saved_img_size}")
    if 'epoch' in checkpoint:
        print(f"   - 训练轮数: {checkpoint['epoch']}")
    if 'best_metric' in checkpoint:
        print(f"   - 最佳指标: {checkpoint['best_metric']:.6f}")
    
    # 使用checkpoint中的配置（如果有）
    if saved_img_size != img_size:
        print(f"   ⚠️  使用checkpoint中的图像尺寸: {saved_img_size}")
        img_size = saved_img_size
    
    if saved_arch != arch:
        print(f"   ⚠️  使用checkpoint中的模型架构: {saved_arch}")
        arch = saved_arch
    
    # 2. 重建DataLoaders（非分布式）
    print(f"\n📊 构建数据集...")
    print(f"   数据路径: {data_path}")
    print(f"   图像尺寸: {img_size}")
    print(f"   Batch大小: {batch_size}")
    
    dls = ImageDataLoaders.from_folder(
        data_path,
        valid='val',
        bs=batch_size,
        item_tfms=Resize(img_size),
        batch_tfms=None,  # 评估不需要数据增强
        num_workers=min(8, os.cpu_count()),
        shuffle=False
    )
    
    print(f"   训练集: {len(dls.train.dataset)} 样本")
    print(f"   验证集: {len(dls.valid.dataset)} 样本")
    print(f"   类别数: {len(dls.vocab)}")
    print(f"   类别: {dls.vocab[:10]}{'...' if len(dls.vocab) > 10 else ''}")
    
    # 3. 创建Learner
    print(f"\n🔧 创建Learner...")
    print(f"   架构: {arch}")
    
    learn = vision_learner(
        dls,
        arch=arch,
        metrics=[
            accuracy,
            error_rate,
            Precision(average='weighted'),
            Recall(average='weighted'),
            F1Score(average='weighted')
        ],
        pretrained=False  # 不需要预训练权重，我们会加载训练好的权重
    )
    
    # 关键修复：移除所有分布式相关的callbacks
    # 评估脚本作为独立进程运行，不需要分布式同步
    print(f"\n🔧 移除分布式callbacks（独立进程模式）...")
    from fastai.distributed import DistributedTrainer, GatherPredsCallback
    
    # 移除可能存在的分布式callbacks
    callbacks_to_remove = []
    for cb in learn.cbs:
        if isinstance(cb, (DistributedTrainer, GatherPredsCallback)):
            callbacks_to_remove.append(cb)
            print(f"   - 移除: {cb.__class__.__name__}")
    
    for cb in callbacks_to_remove:
        learn.remove_cb(cb)
    
    if not callbacks_to_remove:
        print(f"   ✅ 未检测到分布式callbacks")
    
    # 4. 加载权重
    print(f"\n⚙️  加载模型权重...")
    model_state = checkpoint['model']
    
    # 处理可能的DDP前缀
    if any(k.startswith('module.') for k in model_state.keys()):
        print(f"   检测到DDP前缀，正在移除...")
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    learn.model.load_state_dict(model_state)
    print(f"   ✅ 模型权重加载成功")
    
    print(f"{'='*80}\n")
    
    return learn

def evaluate_with_learner(
    learn,
    model_path,
    output_dir='./evaluation_results',
    mlflow_run_id=None
):
    """
    使用已有的Learner对象进行评估（复用训练数据，避免重复加载）
    
    Args:
        learn: FastAI Learner对象（已训练好的）
        model_path: 模型checkpoint路径（用于加载best权重）
        output_dir: 输出目录
        mlflow_run_id: MLflow Run ID（可选，用于关联训练Run）
    """
    
    # 关键修复：清理分布式相关的环境变量
    import os
    distributed_env_vars = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 
        'MASTER_ADDR', 'MASTER_PORT',
        'TORCH_DISTRIBUTED_DEBUG'
    ]
    
    cleaned_vars = []
    for var in distributed_env_vars:
        if var in os.environ:
            del os.environ[var]
            cleaned_vars.append(var)
    
    if cleaned_vars:
        print(f"🔧 清理分布式环境变量: {', '.join(cleaned_vars)}")
    
    print(f"\n{'='*80}")
    print("准备评估（复用训练数据）")
    print(f"{'='*80}")
    
    # 移除分布式callbacks
    print(f"\n🔧 移除分布式callbacks（独立评估模式）...")
    from fastai.distributed import DistributedTrainer, GatherPredsCallback
    
    callbacks_to_remove = []
    for cb in learn.cbs:
        if isinstance(cb, (DistributedTrainer, GatherPredsCallback)):
            callbacks_to_remove.append(cb)
            print(f"   - 移除: {cb.__class__.__name__}")
    
    for cb in callbacks_to_remove:
        learn.remove_cb(cb)
    
    if not callbacks_to_remove:
        print(f"   ✅ 未检测到分布式callbacks")
    
    # 加载best模型权重
    model_path = Path(model_path)
    if model_path.exists():
        print(f"\n📦 加载最佳模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model_state = checkpoint['model']
        
        # 处理可能的DDP前缀
        if any(k.startswith('module.') for k in model_state.keys()):
            print(f"   检测到DDP前缀，正在移除...")
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
        
        learn.model.load_state_dict(model_state)
        print(f"   ✅ 模型权重加载成功")
    else:
        print(f"   ⚠️  模型文件不存在，使用当前权重: {model_path}")
    
    # 执行评估
    print(f"\n{'='*80}")
    print("开始评估")
    print(f"{'='*80}")
    print("正在对验证集进行预测...")
    print(f"   数据集: {len(learn.dls.valid.dataset)} 个样本")
    print(f"   类别数: {len(learn.dls.vocab)}")
    
    preds, targs = learn.get_preds(dl=learn.dls.valid)
    
    print(f"✅ 预测完成 ({len(targs)} 个样本)")
    
    # 生成报告
    generate_reports(
        learn=learn,
        preds=preds,
        targs=targs,
        output_dir=output_dir,
        mlflow_run_id=mlflow_run_id
    )
    
    print(f"\n{'='*80}")
    print(f"✅ 评估完成！结果保存在: {Path(output_dir).absolute()}")
    print(f"{'='*80}\n")

def evaluate_model(
    model_path,
    data_path,
    img_size=224,
    batch_size=256,
    arch='resnet18',
    output_dir='./evaluation_results',
    mlflow_run_id=None
):
    """
    执行模型评估
    
    Args:
        model_path: 模型checkpoint路径
        data_path: 数据集路径
        img_size: 图像尺寸
        batch_size: batch大小
        arch: 模型架构
        output_dir: 输出目录
        mlflow_run_id: MLflow Run ID（可选，用于关联训练Run）
    """
    
    # 1. 加载模型
    learn = load_model_from_checkpoint(
        checkpoint_path=model_path,
        data_path=data_path,
        img_size=img_size,
        batch_size=batch_size,
        arch=arch
    )
    
    # 2. 执行评估
    print(f"{'='*80}")
    print("开始评估")
    print(f"{'='*80}")
    print("正在对验证集进行预测...")
    
    preds, targs = learn.get_preds(dl=learn.dls.valid)
    
    print(f"✅ 预测完成 ({len(targs)} 个样本)")
    
    # 3. 生成报告
    generate_reports(
        learn=learn,
        preds=preds,
        targs=targs,
        output_dir=output_dir,
        mlflow_run_id=mlflow_run_id
    )
    
    print(f"\n{'='*80}")
    print(f"✅ 评估完成！结果保存在: {Path(output_dir).absolute()}")
    print(f"{'='*80}\n")

def generate_reports(learn, preds, targs, output_dir, mlflow_run_id=None):
    """
    生成评估报告
    
    Args:
        learn: Learner对象
        preds: 预测结果
        targs: 真实标签
        output_dir: 输出目录
        mlflow_run_id: MLflow Run ID（可选，用于上传结果）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = learn.dls.vocab
    
    # 转换标签
    targ_indices = targs.cpu().numpy()
    pred_indices = preds.argmax(dim=1).cpu().numpy()
    classes_array = np.array(classes)
    true_labels = classes_array[targ_indices].tolist()
    pred_labels = classes_array[pred_indices].tolist()
    
    # 1. 类别预测统计
    print_class_statistics(classes, true_labels, pred_labels)
    
    # 2. 混淆矩阵
    save_confusion_matrix(
        targ_indices, pred_indices, classes, 
        output_dir, mlflow_run_id
    )
    
    # 3. 分类报告
    save_classification_report(
        true_labels, pred_labels, classes,
        output_dir, mlflow_run_id
    )

def print_class_statistics(classes, true_labels, pred_labels):
    """打印类别预测统计"""
    print("\n" + "="*80)
    print("类别预测统计")
    print("="*80)
    print(f"{'类别':<20} {'真实样本数':<15} {'被预测次数':<15} {'状态':<20}")
    print("-"*80)
    
    true_label_counts = pd.Series(true_labels).value_counts()
    pred_label_counts = pd.Series(pred_labels).value_counts()
    
    classes_with_no_predictions = []
    classes_with_no_samples = []
    
    for class_name in classes:
        true_count = true_label_counts.get(class_name, 0)
        pred_count = pred_label_counts.get(class_name, 0)
        
        if pred_count == 0 and true_count > 0:
            status = "⚠️  未被预测到"
            classes_with_no_predictions.append(class_name)
        elif true_count == 0 and pred_count > 0:
            status = "⚠️  误预测（无真实样本）"
            classes_with_no_samples.append(class_name)
        elif true_count == 0 and pred_count == 0:
            status = "❓ 无数据"
        else:
            status = "✓ 正常"
        
        print(f"{class_name:<20} {true_count:<15} {pred_count:<15} {status:<20}")
    
    print("-"*80)
    
    if classes_with_no_predictions:
        print(f"\n⚠️  警告：以下类别在验证集中有样本但未被模型预测到：")
        for cls in classes_with_no_predictions:
            print(f"   - {cls} (真实样本数: {true_label_counts[cls]})")
        print(f"   这些类别的 Precision 将被设为 0.0")
    
    if classes_with_no_samples:
        print(f"\n⚠️  警告：以下类别在验证集中无真实样本但被模型误预测：")
        for cls in classes_with_no_samples:
            print(f"   - {cls} (被预测次数: {pred_label_counts[cls]})")
        print(f"   这些类别的 Recall 将被设为 0.0")
    
    print("="*80 + "\n")

def save_confusion_matrix(targ_indices, pred_indices, classes, output_dir, mlflow_run_id=None):
    """保存混淆矩阵"""
    print("绘制混淆矩阵...")
    
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(targ_indices, pred_indices)
        
        # 归一化混淆矩阵
        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar=True)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 保存到文件
        normalized_path = output_dir / 'confusion_matrix_normalized.png'
        plt.savefig(normalized_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ 保存归一化混淆矩阵: {normalized_path}")
        
        # 上传到 MLflow
        if mlflow_run_id:
            upload_figure_to_mlflow(plt.gcf(), 'confusion_matrix_normalized', mlflow_run_id)
        
        plt.close()
        
        # 原始计数混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar=True)
        plt.title('Confusion Matrix (Count)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 保存到文件
        count_path = output_dir / 'confusion_matrix_count.png'
        plt.savefig(count_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ 保存计数混淆矩阵: {count_path}")
        
        # 上传到 MLflow
        if mlflow_run_id:
            upload_figure_to_mlflow(plt.gcf(), 'confusion_matrix_count', mlflow_run_id)
        
        plt.close()
        
        # 最容易混淆的类别对
        print("\n最容易混淆的类别对:")
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((classes[i], classes[j], cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        if confused_pairs:
            for actual, predicted, count in confused_pairs:
                print(f"   真实: {actual:<15} → 预测为: {predicted:<15} (错误 {int(count)} 次)")
        else:
            print("   无明显混淆")
        
    except Exception as e:
        print(f"⚠️  绘制混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()

def save_classification_report(true_labels, pred_labels, classes, output_dir, mlflow_run_id=None):
    """保存分类报告"""
    print("\n生成详细分类报告...")
    
    # 生成报告
    report = classification_report(
        true_labels,
        pred_labels,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
    
    # 打印报告
    print("\n" + "="*80)
    print("分类报告 (Classification Report)")
    print("="*80)
    print(classification_report(
        true_labels,
        pred_labels,
        labels=classes,
        zero_division=0
    ))
    
    # 保存为DataFrame
    report_df = pd.DataFrame({
        'class': classes,
        'precision': [report[c]['precision'] if c in report else 0 for c in classes],
        'recall': [report[c]['recall'] if c in report else 0 for c in classes],
        'f1-score': [report[c]['f1-score'] if c in report else 0 for c in classes],
        'support': [report[c]['support'] if c in report else 0 for c in classes]
    })
    
    # 添加总体指标
    overall_metrics = pd.DataFrame({
        'class': ['accuracy', 'macro avg', 'weighted avg'],
        'precision': [
            report.get('accuracy', 0),
            report['macro avg']['precision'],
            report['weighted avg']['precision']
        ],
        'recall': [
            report.get('accuracy', 0),
            report['macro avg']['recall'],
            report['weighted avg']['recall']
        ],
        'f1-score': [
            report.get('accuracy', 0),
            report['macro avg']['f1-score'],
            report['weighted avg']['f1-score']
        ],
        'support': [
            sum(report_df['support']),
            sum(report_df['support']),
            sum(report_df['support'])
        ]
    })
    
    full_report_df = pd.concat([report_df, overall_metrics], ignore_index=True)
    
    # 保存到CSV
    csv_path = output_dir / 'classification_report.csv'
    full_report_df.to_csv(csv_path, index=False)
    print(f"   ✅ 保存分类报告: {csv_path}")
    
    # 上传到 MLflow
    if mlflow_run_id:
        upload_metrics_to_mlflow(report, classes, mlflow_run_id)
        upload_artifact_to_mlflow(csv_path, mlflow_run_id)
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='评估训练好的模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（仅保存本地文件）
  python fastai/evaluate.py --model best.pth --data /mnt/ssd/dataset
  
  # 关联到 MLflow 训练 Run
  python fastai/evaluate.py \\
      --model runs/ai-classifier/resnet18/best.pth \\
      --data /mnt/ssd/dataset \\
      --img_size 224 \\
      --batch_size 256 \\
      --arch resnet18 \\
      --output_dir ./evaluation_results \\
      --mlflow_run_id <run_id>
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型checkpoint路径 (必需)')
    parser.add_argument('--data', type=str, required=True,
                       help='数据集路径 (必需)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='输入图像尺寸 (默认: 224)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch大小 (默认: 256)')
    parser.add_argument('--arch', type=str, default='resnet18',
                       help='模型架构 (默认: resnet18)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='输出目录 (默认: ./evaluation_results)')
    parser.add_argument('--mlflow_run_id', type=str, default=None,
                       help='MLflow Run ID (可选，用于关联训练Run)')
    
    args = parser.parse_args()
    
    try:
        evaluate_model(
            model_path=args.model,
            data_path=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            arch=args.arch,
            output_dir=args.output_dir,
            mlflow_run_id=args.mlflow_run_id
        )
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
