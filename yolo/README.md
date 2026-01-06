# YOLOv11 检测模型训练 - MLflow 集成

统一的 YOLOv11 目标检测训练脚本，集成 MLflow 实验追踪，实现训练、评估、可视化的全流程自动化。

---

## 🚀 快速开始

### 1. 训练模型

```bash
python yolo/train.py \
    --data pk-dataset.yaml \
    --model yolo11s.pt \
    --epochs 100 \
    --batch 16 \
    --project_name cell-detection \
    --task_name yolo11s-exp1
```

### 2. 评估模型

```bash
python yolo/evaluate.py \
    --model runs/detect/yolo11s-exp1/weights/best.pt \
    --data pk-dataset.yaml \
    --mlflow_run_id <run_id>
```

### 3. 可视化结果

```bash
python yolo/visualize.py \
    --model runs/detect/yolo11s-exp1/weights/best.pt \
    --image_dir /data/test_images \
    --output_dir visualizations
```

---

## 📋 功能特性

### ✅ 统一训练脚本
- 使用 YOLO 官方 API 训练
- MLflow 自动记录参数和指标
- 自动上传模型到 Model Registry
- 支持断点续训
- 支持所有 YOLO 训练参数

### ✅ 自动评估
- 计算 mAP@0.5、mAP@0.5:0.95
- 计算 Precision、Recall、F1-Score
- 生成混淆矩阵
- 自动上传评估结果

### ✅ 检测可视化
- 自动在图像上绘制边界框
- 支持批量处理
- 自动上传示例图像到 MLflow

### ✅ MLflow 整合
- 实时记录训练指标
- 模型版本管理
- 实验对比和追踪
- 完整的可复现性

---

## 📊 MLflow 记录内容

### 参数 (Parameters)
```
model, epochs, imgsz, batch, data_yaml
dataset/num_classes, dataset/classes
optimizer, lr0, lrf, momentum
数据增强参数（hsv_h, hsv_s, mosaic, mixup 等）
```

### 指标 (Metrics)
```
final/mAP50, final/mAP50-95
final/precision, final/recall
final/box_loss, final/cls_loss, final/dfl_loss
eval/mAP50, eval/precision, eval/recall, eval/f1
```

### 文件 (Artifacts)
```
models/
  ├── best.pt
  └── last.pt

plots/
  ├── confusion_matrix.png
  ├── results.png
  ├── PR_curve.png
  ├── F1_curve.png
  └── labels.jpg

training_results/
  └── results.csv

evaluation/
  └── evaluation_metrics.csv

visualizations/
  ├── image1_pred.jpg
  └── ...
```

---

## 💡 使用示例

### 示例 1: 基础训练

```bash
python yolo/train.py \
    --data pk-dataset.yaml \
    --model yolo11s.pt \
    --epochs 100
```

### 示例 2: 完整配置训练

```bash
python yolo/train.py \
    --data dna-classify-cluster.yaml \
    --model yolo11m.pt \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --project_name dna-detection \
    --task_name yolo11m-high-quality \
    --device 0 \
    --workers 16 \
    --patience 100 \
    --optimizer AdamW \
    --lr0 0.001 \
    --mosaic 1.0 \
    --mixup 0.1
```

### 示例 3: 断点续训

```bash
python yolo/train.py \
    --data pk-dataset.yaml \
    --model runs/detect/yolo11s-exp1/weights/last.pt \
    --epochs 200 \
    --resume
```

### 示例 4: 评估特定 run

```bash
# 获取 MLflow Run ID from UI 或训练输出
RUN_ID="abc123..."

python yolo/evaluate.py \
    --model runs/detect/yolo11s-exp1/weights/best.pt \
    --data pk-dataset.yaml \
    --mlflow_run_id $RUN_ID
```

### 示例 5: 可视化检测结果

```bash
python yolo/visualize.py \
    --model runs/detect/yolo11s-exp1/weights/best.pt \
    --image_dir /mnt/ssd/test_images \
    --output_dir my_predictions \
    --conf 0.5 \
    --max_images 100
```

---

## ⚙️ 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | 必需 | 数据集 YAML 配置文件 |
| `--model` | yolo11s.pt | YOLO 模型 |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--batch` | 16 | 批次大小 |
| `--project_name` | yolo-detection | MLflow 项目名称 |
| `--task_name` | experiment | MLflow 运行名称 |
| `--device` | 0 | GPU 设备 |
| `--workers` | 8 | 数据加载线程数 |
| `--patience` | 50 | 早停轮数 |
| `--optimizer` | auto | 优化器 (SGD/Adam/AdamW/auto) |
| `--lr0` | 0.01 | 初始学习率 |
| `--mosaic` | 1.0 | Mosaic 增强概率 |
| `--mixup` | 0.0 | Mixup 增强概率 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | 模型路径 |
| `--data` | 必需 | 数据集配置 |
| `--mlflow_run_id` | None | MLflow Run ID（可选） |
| `--conf` | 0.001 | 置信度阈值 |
| `--iou` | 0.7 | NMS IoU 阈值 |

### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | 模型路径 |
| `--image_dir` | 必需 | 图像目录 |
| `--output_dir` | visualizations | 输出目录 |
| `--conf` | 0.25 | 置信度阈值 |
| `--max_images` | 20 | 最大图像数 |

---

## 📁 数据集格式

YOLOv11 需要以下数据集格式：

### dataset.yaml

```yaml
path: /path/to/dataset  # 数据集根目录
train: images/train     # 训练图像相对路径
val: images/val         # 验证图像相对路径

nc: 5  # 类别数
names:
  0: class1
  1: class2
  2: class3
  3: class4
  4: class5
```

### 目录结构

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── img1.txt
        └── ...
```

### 标注格式 (YOLO 格式)

每个图像对应一个 .txt 文件，每行一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

坐标归一化到 [0, 1]:
- `x_center`, `y_center`: 边界框中心相对于图像宽高的比例
- `width`, `height`: 边界框宽高相对于图像宽高的比例

示例:
```
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25
```

---

## 🌐 MLflow UI

训练完成后，访问 MLflow UI 查看结果：

```
http://192.168.16.130:5000/
```

功能:
- 查看所有实验和运行
- 对比不同配置的性能
- 下载模型和结果
- 查看可视化图表
- 管理模型版本

---

## ⚠️ 注意事项

### 1. 计算资源

推荐配置:
- **GPU**: RTX 3090 或更好
- **内存**: 至少 16GB RAM
- **存储**: 至少 50GB 空闲空间

预计训练时间（RTX 3090）:
- yolo11n: ~1000 张 → 0.5-1 小时
- yolo11s: ~1000 张 → 1-2 小时
- yolo11m: ~5000 张 → 4-6 小时

### 2. MLflow 存储

模型文件较大（100MB-500MB），建议配置:
- 使用 MinIO 或 S3 作为 artifact store
- 定期清理旧的实验

### 3. 数据集准备

确保数据集:
- ✅ 标注格式正确（YOLO 格式）
- ✅ 图像和标注文件名匹配
- ✅ 类别 ID 在 [0, nc-1] 范围内
- ✅ 坐标归一化到 [0, 1]

---

## 🔗 相关链接

- [YOLOv11 官方文档](https://docs.ultralytics.com/models/yolov11/)
- [MLflow 官方文档](https://mlflow.org/docs/latest/index.html)
- [项目 MLflow 文档](../README_MLFLOW.md)
- [完整提案](../openspec/specs/yolov11-detection-mlflow.md)

---

## 📝 更新日志

### v1.0 (2025-11-12)
- ✅ 初始版本
- ✅ 训练脚本 + MLflow 集成
- ✅ 评估脚本
- ✅ 可视化脚本
- ✅ 完整文档

---

**创建日期**: 2025-11-12  
**版本**: v1.0  
**状态**: ✅ 已实施

## 超参数调优

YOLO 模型支持自动化超参数调优，使用与 FastAI 相同的基础设施。

### 快速开始

1. **创建配置文件** (或使用示例配置):
```bash
configs/tuning/yolo-detection.yaml
```

2. **运行调优** (单GPU):
```bash
python yolo/tune.py --config configs/tuning/yolo-detection.yaml
```

3. **分布式调优** (多GPU):
```bash
torchrun --nproc_per_node=4 yolo/tune.py \
  --config configs/tuning/yolo-detection.yaml \
  --distributed
```

4. **预览参数组合** (不训练):
```bash
python yolo/tune.py --config configs/tuning/yolo-detection.yaml --dry-run
```

### 配置说明

```yaml
strategy: optuna  # 搜索策略: grid, random, optuna
n_trials: 15      # 试验次数

metric: metrics/mAP50-95(B)  # 优化指标
mode: maximize                # maximize 或 minimize

search_space:
  lr0:              # 学习率
    type: log_uniform
    min: 0.0001
    max: 0.01
  
  batch:            # 批次大小
    type: choice
    values: [8, 16, 32]
  
  # ... 更多参数

base_args:
  data_yaml: datasets/your-dataset/data.yaml
  model: yolo11n.pt
  epochs: 100
  # ... 固定参数
```

### MLflow 集成

所有调优运行自动记录到 MLflow，采用嵌套结构：

```
📁 Experiment: your-project
  └─ 📄 Parent Run: yolo11n-optuna
       ├─ 📄 trial_000_lr0.001_batch16_img640
       ├─ 📄 trial_001_lr0.005_batch32_img800
       └─ ... (所有 trials)
```

查看结果：http://192.168.16.130:5000/

### 支持的搜索策略

- **Grid Search**: 穷举所有组合
- **Random Search**: 随机采样
- **Optuna**: 贝叶斯优化 (推荐)

详细文档请参考: `docs/yolo-tuning.md`
