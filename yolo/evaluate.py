"""YOLOv11 检测模型评估"""
import mlflow
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import argparse

def evaluate_yolo(
    model_path: str,
    data_yaml: str,
    mlflow_run_id: str = None,
    mlflow_uri: str = 'http://192.168.16.130:5000/',
    save_dir: str = 'evaluation_results',
    **kwargs
):
    """
    评估 YOLOv11 检测模型
    
    Args:
        model_path: 模型路径
        data_yaml: 数据集配置
        mlflow_run_id: MLflow Run ID（用于继续记录）
        mlflow_uri: MLflow 服务器地址
        save_dir: 结果保存目录
        **kwargs: 其他评估参数
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"📊 YOLOv11 检测模型评估")
    print(f"{'='*80}")
    print(f"模型: {model_path}")
    print(f"数据集: {data_yaml}")
    print(f"结果目录: {save_dir}")
    if mlflow_run_id:
        print(f"MLflow Run ID: {mlflow_run_id}")
    print(f"{'='*80}\n")
    
    # 加载模型
    print(f"🔧 加载模型...")
    model = YOLO(model_path)
    
    # 评估
    print(f"\n🏃 开始评估...\n")
    metrics = model.val(
        data=data_yaml,
        save_json=True,
        save_hybrid=True,
        plots=True,
        **kwargs
    )
    
    # 提取指标
    results = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
    }
    
    # 计算 F1-Score
    if results['precision'] > 0 or results['recall'] > 0:
        results['f1'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'] + 1e-6)
    else:
        results['f1'] = 0.0
    
    # 保存结果
    results_df = pd.DataFrame([results])
    results_df.to_csv(save_dir / 'evaluation_metrics.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 评估完成")
    print(f"{'='*80}")
    print(f"   mAP@0.5:      {results['mAP50']:.4f}")
    print(f"   mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    print(f"   Precision:    {results['precision']:.4f}")
    print(f"   Recall:       {results['recall']:.4f}")
    print(f"   F1-Score:     {results['f1']:.4f}")
    print(f"{'='*80}\n")
    
    # 记录到 MLflow（如果提供了 run_id）
    if mlflow_run_id:
        print(f"📊 记录评估结果到 MLflow...")
        mlflow.set_tracking_uri(mlflow_uri)
        
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metrics({
                    'eval/mAP50': results['mAP50'],
                    'eval/mAP50-95': results['mAP50-95'],
                    'eval/precision': results['precision'],
                    'eval/recall': results['recall'],
                    'eval/f1': results['f1'],
                })
                mlflow.log_artifact(str(save_dir / 'evaluation_metrics.csv'), 'evaluation')
                print(f"   ✅ 评估结果已上传到 MLflow")
        except Exception as e:
            print(f"   ⚠️  上传到 MLflow 失败: {e}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLOv11 检测模型评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python yolo/evaluate.py \\
      --model runs/detect/exp/weights/best.pt \\
      --data pk-dataset.yaml

  python yolo/evaluate.py \\
      --model runs/detect/exp/weights/best.pt \\
      --data pk-dataset.yaml \\
      --mlflow_run_id <run_id>
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--data', type=str, required=True,
                       help='数据集 YAML 配置文件')
    parser.add_argument('--mlflow_run_id', type=str, default=None,
                       help='MLflow Run ID（可选）')
    parser.add_argument('--mlflow_uri', type=str,
                       default='http://192.168.16.130:5000/',
                       help='MLflow 服务器地址')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='结果保存目录')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='NMS IoU 阈值')
    
    args = parser.parse_args()
    
    evaluate_yolo(
        model_path=args.model,
        data_yaml=args.data,
        mlflow_run_id=args.mlflow_run_id,
        mlflow_uri=args.mlflow_uri,
        save_dir=args.save_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
    )
