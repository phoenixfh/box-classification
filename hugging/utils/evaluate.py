"""
Evaluation utilities for HuggingFace training

Provides functions for generating confusion matrices, classification reports,
and prediction statistics after training completion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from typing import Optional, List
import mlflow

from .logger import print_main


def generate_evaluation_reports(
    predictions,
    targets, 
    classes: List[str],
    output_dir: Path,
    mlflow_run_id: Optional[str] = None
):
    """
    Generate comprehensive evaluation reports including confusion matrix and classification report.
    
    Args:
        predictions: Model predictions (logits or probabilities), shape (N, num_classes)
        targets: Ground truth labels, shape (N,)
        classes: List of class names
        output_dir: Directory to save evaluation artifacts
        mlflow_run_id: Optional MLflow run ID for uploading artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy arrays
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    
    # Get predicted class indices
    pred_indices = predictions.argmax(axis=1)
    targ_indices = targets
    
    # Convert to class names
    classes_array = np.array(classes)
    true_labels = classes_array[targ_indices].tolist()
    pred_labels = classes_array[pred_indices].tolist()
    
    # Generate all reports
    print_main("\n" + "="*80)
    print_main("📊 生成评估报告")
    print_main("="*80)
    
    # 1. Class statistics
    print_class_statistics(classes, true_labels, pred_labels)
    
    # 2. Confusion matrix
    save_confusion_matrix(
        targ_indices, pred_indices, classes,
        output_dir, mlflow_run_id
    )
    
    # 3. Classification report
    save_classification_report(
        true_labels, pred_labels, classes,
        output_dir, mlflow_run_id
    )
    
    print_main(f"\n✅ 评估报告已保存到: {output_dir.absolute()}")
    print_main("="*80 + "\n")


def print_class_statistics(classes: List[str], true_labels: List[str], pred_labels: List[str]):
    """
    Print class prediction statistics showing distribution of true vs predicted samples.
    
    Args:
        classes: List of all class names
        true_labels: List of true class labels
        pred_labels: List of predicted class labels
    """
    print_main("\n" + "="*80)
    print_main("类别预测统计")
    print_main("="*80)
    print_main(f"{'类别':<20} {'真实样本数':<15} {'被预测次数':<15} {'状态':<20}")
    print_main("-"*80)
    
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
        
        print_main(f"{class_name:<20} {true_count:<15} {pred_count:<15} {status:<20}")
    
    print_main("-"*80)
    
    if classes_with_no_predictions:
        print_main(f"\n⚠️  警告：以下类别在验证集中有样本但未被模型预测到：")
        for cls in classes_with_no_predictions:
            print_main(f"   - {cls} (真实样本数: {true_label_counts[cls]})")
        print_main(f"   这些类别的 Precision 将被设为 0.0")
    
    if classes_with_no_samples:
        print_main(f"\n⚠️  警告：以下类别在验证集中无真实样本但被模型误预测：")
        for cls in classes_with_no_samples:
            print_main(f"   - {cls} (被预测次数: {pred_label_counts[cls]})")
        print_main(f"   这些类别的 Recall 将被设为 0.0")
    
    print_main("="*80 + "\n")


def save_confusion_matrix(
    targ_indices: np.ndarray,
    pred_indices: np.ndarray,
    classes: List[str],
    output_dir: Path,
    mlflow_run_id: Optional[str] = None
):
    """
    Generate and save confusion matrices (normalized and count versions).
    
    Args:
        targ_indices: Ground truth class indices
        pred_indices: Predicted class indices
        classes: List of class names
        output_dir: Directory to save plots
        mlflow_run_id: Optional MLflow run ID for uploading artifacts
    """
    print_main("绘制混淆矩阵...")
    
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(targ_indices, pred_indices)
        
        # Normalized confusion matrix
        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar=True)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save to file
        normalized_path = output_dir / 'confusion_matrix_normalized.png'
        plt.savefig(normalized_path, dpi=150, bbox_inches='tight')
        print_main(f"   ✅ 保存归一化混淆矩阵: {normalized_path}")
        
        # Upload to MLflow
        if mlflow_run_id:
            _upload_figure_to_mlflow(plt.gcf(), 'confusion_matrix_normalized', mlflow_run_id)
        
        plt.close()
        
        # Count confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar=True)
        plt.title('Confusion Matrix (Count)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save to file
        count_path = output_dir / 'confusion_matrix_count.png'
        plt.savefig(count_path, dpi=150, bbox_inches='tight')
        print_main(f"   ✅ 保存计数混淆矩阵: {count_path}")
        
        # Upload to MLflow
        if mlflow_run_id:
            _upload_figure_to_mlflow(plt.gcf(), 'confusion_matrix_count', mlflow_run_id)
        
        plt.close()
        
        # Most confused class pairs
        print_main("\n最容易混淆的类别对:")
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((classes[i], classes[j], cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        if confused_pairs:
            for actual, predicted, count in confused_pairs[:10]:  # Show top 10
                print_main(f"   真实: {actual:<15} → 预测为: {predicted:<15} (错误 {int(count)} 次)")
        else:
            print_main("   无明显混淆")
        
    except Exception as e:
        print_main(f"⚠️  绘制混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()


def save_classification_report(
    true_labels: List[str],
    pred_labels: List[str],
    classes: List[str],
    output_dir: Path,
    mlflow_run_id: Optional[str] = None
):
    """
    Generate and save detailed classification report with per-class metrics.
    
    Args:
        true_labels: Ground truth class labels
        pred_labels: Predicted class labels
        classes: List of all class names
        output_dir: Directory to save report
        mlflow_run_id: Optional MLflow run ID for uploading artifacts
    """
    print_main("\n生成详细分类报告...")
    
    # Generate report
    report = classification_report(
        true_labels,
        pred_labels,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
    
    # Print report to console
    print_main("\n" + "="*80)
    print_main("分类报告 (Classification Report)")
    print_main("="*80)
    print_main(classification_report(
        true_labels,
        pred_labels,
        labels=classes,
        zero_division=0
    ))
    
    # Convert to DataFrame
    report_df = pd.DataFrame({
        'class': classes,
        'precision': [report[c]['precision'] if c in report else 0 for c in classes],
        'recall': [report[c]['recall'] if c in report else 0 for c in classes],
        'f1-score': [report[c]['f1-score'] if c in report else 0 for c in classes],
        'support': [report[c]['support'] if c in report else 0 for c in classes]
    })
    
    # Add overall metrics
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
    
    # Save to CSV
    csv_path = output_dir / 'classification_report.csv'
    full_report_df.to_csv(csv_path, index=False)
    print_main(f"   ✅ 保存分类报告: {csv_path}")
    
    # Upload to MLflow
    if mlflow_run_id:
        _upload_metrics_to_mlflow(report, classes, mlflow_run_id)
        _upload_artifact_to_mlflow(csv_path, mlflow_run_id)
    
    print_main("="*80)


def _upload_figure_to_mlflow(figure, title: str, run_id: str):
    """Upload matplotlib figure to MLflow as artifact."""
    import tempfile
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            figure.savefig(tmp.name, dpi=150, bbox_inches='tight')
            tmp_path = tmp.name
        
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(tmp_path, artifact_path=f"evaluation/{title}.png")
        
        Path(tmp_path).unlink()
        print_main(f"   📤 已上传到 MLflow: {title}")
    except Exception as e:
        print_main(f"   ⚠️  上传 {title} 到 MLflow 失败: {e}")


def _upload_metrics_to_mlflow(report: dict, classes: List[str], run_id: str):
    """Upload per-class metrics to MLflow."""
    try:
        metrics = {}
        for cls in classes:
            if cls in report:
                metrics[f'eval_class_{cls}_precision'] = report[cls]['precision']
                metrics[f'eval_class_{cls}_recall'] = report[cls]['recall']
                metrics[f'eval_class_{cls}_f1'] = report[cls]['f1-score']
        
        # Add overall metrics
        metrics['eval_macro_avg_precision'] = report['macro avg']['precision']
        metrics['eval_macro_avg_recall'] = report['macro avg']['recall']
        metrics['eval_macro_avg_f1'] = report['macro avg']['f1-score']
        metrics['eval_weighted_avg_precision'] = report['weighted avg']['precision']
        metrics['eval_weighted_avg_recall'] = report['weighted avg']['recall']
        metrics['eval_weighted_avg_f1'] = report['weighted avg']['f1-score']
        
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
        
        print_main(f"   📤 已上传 {len(metrics)} 个评估指标到 MLflow")
    except Exception as e:
        print_main(f"   ⚠️  上传指标到 MLflow 失败: {e}")


def _upload_artifact_to_mlflow(file_path: Path, run_id: str):
    """Upload file artifact to MLflow."""
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(file_path), artifact_path="evaluation")
        
        print_main(f"   📤 已上传到 MLflow: {file_path.name}")
    except Exception as e:
        print_main(f"   ⚠️  上传 {file_path.name} 到 MLflow 失败: {e}")
