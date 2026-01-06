"""YOLOv11 检测结果可视化"""
import mlflow
from ultralytics import YOLO
from pathlib import Path
import cv2
import argparse

def visualize_detections(
    model_path: str,
    image_dir: str,
    output_dir: str = 'visualizations',
    conf_threshold: float = 0.25,
    max_images: int = 20,
    mlflow_run_id: str = None,
    mlflow_uri: str = 'http://192.168.16.130:5000/',
):
    """
    可视化检测结果
    
    Args:
        model_path: 模型路径
        image_dir: 图像目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        max_images: 最多可视化图像数
        mlflow_run_id: MLflow Run ID
        mlflow_uri: MLflow 服务器地址
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🎨 YOLOv11 检测可视化")
    print(f"{'='*80}")
    print(f"模型: {model_path}")
    print(f"图像目录: {image_dir}")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"最大图像数: {max_images}")
    print(f"{'='*80}\n")
    
    # 加载模型
    print(f"🔧 加载模型...")
    model = YOLO(model_path)
    
    # 获取图像列表
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(list(Path(image_dir).glob(ext)))
        image_paths.extend(list(Path(image_dir).glob(ext.upper())))
    
    image_paths = image_paths[:max_images]
    
    print(f"📁 找到 {len(image_paths)} 张图像")
    
    if len(image_paths) == 0:
        print(f"⚠️  未找到图像，退出")
        return []
    
    print(f"\n🏃 开始生成可视化...\n")
    
    # 处理每张图像
    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] 处理: {img_path.name}")
        
        try:
            # 检测
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            
            # 保存可视化
            for r in results:
                im_array = r.plot()  # 绘制边界框
                im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
                
                output_path = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
                cv2.imwrite(str(output_path), im)
                
                # 打印检测信息
                boxes = r.boxes
                if len(boxes) > 0:
                    print(f"      检测到 {len(boxes)} 个目标")
                else:
                    print(f"      未检测到目标")
        except Exception as e:
            print(f"      ⚠️  处理失败: {e}")
    
    print(f"\n✅ 生成了 {len(list(output_dir.glob('*')))} 张可视化图像")
    print(f"   保存位置: {output_dir}\n")
    
    # 上传到 MLflow
    if mlflow_run_id:
        print(f"📊 上传可视化到 MLflow...")
        mlflow.set_tracking_uri(mlflow_uri)
        
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                # 上传前几张作为示例
                uploaded = 0
                for img in list(output_dir.glob('*'))[:10]:  # 最多上传10张
                    mlflow.log_artifact(str(img), 'visualizations')
                    uploaded += 1
                print(f"   ✅ 上传了 {uploaded} 张示例图像到 MLflow")
        except Exception as e:
            print(f"   ⚠️  上传到 MLflow 失败: {e}")
    
    return list(output_dir.glob('*'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLOv11 检测结果可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python yolo_detection/visualize.py \\
      --model runs/detect/exp/weights/best.pt \\
      --image_dir /data/test_images

  python yolo_detection/visualize.py \\
      --model runs/detect/exp/weights/best.pt \\
      --image_dir /data/test_images \\
      --output_dir my_visualizations \\
      --conf 0.5 \\
      --max_images 50
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像目录')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--max_images', type=int, default=20,
                       help='最多可视化图像数')
    parser.add_argument('--mlflow_run_id', type=str, default=None,
                       help='MLflow Run ID（可选）')
    parser.add_argument('--mlflow_uri', type=str,
                       default='http://192.168.16.130:5000/',
                       help='MLflow 服务器地址')
    
    args = parser.parse_args()
    
    visualize_detections(
        model_path=args.model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        max_images=args.max_images,
        mlflow_run_id=args.mlflow_run_id,
        mlflow_uri=args.mlflow_uri,
    )
