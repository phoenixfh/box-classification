# 宫颈细胞识别训练指南

## 概述

本文档介绍了针对宫颈细胞识别任务优化的 YOLOv11 训练脚本的使用方法。

## 新增功能

### 1. 模型规模选择 (`--model_size`)

通过 `--model_size` 参数快速选择不同规模的模型：

| 参数值 | 对应模型 | 适用场景 |
|--------|----------|----------|
| `nano` | yolo11n.pt | 快速实验、资源受限 |
| `small` | yolo11s.pt | 平衡速度与精度 |
| `medium` | yolo11m.pt | 推荐用于细胞识别 |
| `large` | yolo11l.pt | 高精度要求 |
| `xlarge` | yolo11x.pt | 极致精度 |

**注意：** `--model_size` 会覆盖 `--model` 参数。

### 2. 高级数据增强 (`--use_advanced_aug`)

启用针对细胞图像优化的数据增强策略：

#### 色彩增强
- **hsv_h=0.015**: 模拟细胞染色的色调变化
- **hsv_s=0.7**: 饱和度变化，适应不同染色强度
- **hsv_v=0.4**: 明度变化，适应不同光照条件

#### 几何变换
- **degrees=0.0**: 旋转（细胞无明显方向性）
- **translate=0.1**: 平移增强
- **scale=0.5**: 缩放增强，模拟细胞大小差异
- **flipud=0.5**: 垂直翻转
- **fliplr=0.5**: 水平翻转

#### 混合增强
- **mosaic=1.0**: Mosaic 增强（4张图像拼接）
- **mixup=0.1**: MixUp 增强（图像混合）
- **copy_paste=0.1**: 复制粘贴增强（细胞实例复制）

### 3. 困难样本挖掘 (`--use_hard_mining`)

启用困难样本挖掘策略，提升模型对难检测细胞的识别能力：

#### Focal Loss
- **fl_gamma=2.0**: 使用 Focal Loss 自动关注困难样本

#### 损失权重优化
- **box=7.5**: 边框定位损失权重（提高定位精度）
- **cls=0.5**: 分类损失权重
- **dfl=1.5**: Distribution Focal Loss 权重（细化边框）

#### 优化器配置
- **optimizer=AdamW**: 使用 AdamW 优化器，对细节更敏感
- **lr0=0.001**: 初始学习率
- **lrf=0.01**: 最终学习率系数（衰减到初始的1%）
- **weight_decay=0.0005**: 权重衰减，防止过拟合

#### 学习率调度
- **cos_lr=True**: 余弦学习率衰减
- **warmup_epochs=3.0**: 预热轮数
- **warmup_momentum=0.8**: 预热阶段动量
- **warmup_bias_lr=0.1**: 偏置项预热学习率

#### 检测阈值
- **conf=0.25**: 置信度阈值
- **iou=0.7**: NMS IoU阈值
- **patience=100**: 早停耐心值（提高到100轮）

### 4. 通用细胞识别优化

自动启用的优化配置：

- **close_mosaic=10**: 最后10个epoch关闭Mosaic增强，提高收敛精度
- **amp=True**: 混合精度训练，加速训练并节省显存
- **multi_scale=True**: 多尺度训练，提高对不同尺寸细胞的检测能力
- **save_period=-1**: 只保存best和last模型，节省磁盘空间

## 使用示例

### 推荐配置（高精度训练）

```bash
python yolo/train.py \
    --data docs/cell-core/cell.yaml \
    --model yolo11n.pt \
    --epochs 5000 \
    --imgsz 1024 \
    --batch 8 \
    --device 0,1,2,3,4 \
    --model_size medium \
    --use_advanced_aug \
    --use_hard_mining \
    --project_name cell-box-modify \
    --task_name high-accuracy
```

**说明：**
- 使用 1024x1024 输入尺寸，捕捉细胞细节
- 5000 轮训练，确保充分学习
- 5块GPU，每块batch=8（总batch=40）
- medium规模模型（yolo11m.pt）
- 启用所有细胞识别优化

### 快速实验

```bash
python yolo/train.py \
    --data docs/cell-core/cell.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --model_size nano \
    --project_name cell-quick-test \
    --task_name test-001
```

### 仅使用高级数据增强

```bash
python yolo/train.py \
    --data docs/cell-core/cell.yaml \
    --epochs 1000 \
    --imgsz 1024 \
    --batch 8 \
    --device 0,1 \
    --model_size medium \
    --use_advanced_aug \
    --project_name cell-aug-only \
    --task_name exp-001
```

### 仅使用困难样本挖掘

```bash
python yolo/train.py \
    --data docs/cell-core/cell.yaml \
    --epochs 1000 \
    --imgsz 1024 \
    --batch 8 \
    --device 0,1 \
    --model_size medium \
    --use_hard_mining \
    --project_name cell-mining-only \
    --task_name exp-001
```

## 参数说明

### 必需参数
- `--data`: 数据集YAML配置文件路径

### 基础参数
- `--model`: YOLO模型文件（会被 `--model_size` 覆盖）
- `--epochs`: 训练轮数
- `--imgsz`: 输入图像尺寸（宫颈细胞推荐1024）
- `--batch`: 单GPU的batch大小（总batch会自动乘以GPU数量）

### 宫颈细胞专用参数
- `--model_size`: 模型规模选择（nano/small/medium/large/xlarge）
- `--use_advanced_aug`: 启用高级数据增强
- `--use_hard_mining`: 启用困难样本挖掘

### MLflow参数
- `--project_name`: MLflow项目名称
- `--task_name`: MLflow运行名称
- `--mlflow_uri`: MLflow服务器地址

### 其他参数
- `--device`: GPU设备（如 `0` 或 `0,1,2,3`）
- `--overwrite`: 强制重新训练（忽略checkpoint）
- `--workers`: 数据加载线程数
- 以及所有YOLO原生支持的训练参数

## 训练监控

### MLflow集成

训练过程会自动记录到MLflow：

1. **实时指标**: 每个epoch的训练和验证指标
2. **最终指标**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
3. **模型文件**: best.pt, last.pt, best.onnx
4. **可视化图表**: 混淆矩阵、PR曲线、F1曲线等

### 查看训练结果

```bash
# 本地目录
runs/cell-box-modify/high-accuracy/

# MLflow UI
# 访问: http://192.168.16.130:5000
```

## 断点续训

脚本支持自动断点续训：

```bash
# 第一次训练
python yolo/train.py --data cell.yaml --epochs 1000 --project_name test --task_name exp1

# 如果中断，直接再次运行相同命令即可恢复
python yolo/train.py --data cell.yaml --epochs 1000 --project_name test --task_name exp1

# 如果想继续训练更多轮次，增加 epochs
python yolo/train.py --data cell.yaml --epochs 2000 --project_name test --task_name exp1
```

## Batch Size 计算

**重要：** `--batch` 参数表示**单个GPU**的batch大小。

总batch大小 = `--batch` × GPU数量

示例：
```bash
# 单GPU: batch=16
python train.py --data cell.yaml --batch 16 --device 0
# 总batch = 16

# 多GPU: batch=8, 5个GPU
python train.py --data cell.yaml --batch 8 --device 0,1,2,3,4
# 总batch = 8 × 5 = 40
# 每个GPU实际batch = 8
```

## 性能优化建议

### 显存不足
1. 减小 `--batch` 值
2. 减小 `--imgsz` 值
3. 使用更小的模型（nano/small）

### 训练速度慢
1. 增加 `--workers` 值（默认8）
2. 使用更快的存储（SSD）
3. 使用更多GPU

### 精度提升
1. 增加 `--epochs`
2. 增加 `--imgsz`（如1024）
3. 启用 `--use_advanced_aug`
4. 启用 `--use_hard_mining`
5. 使用更大的模型（medium/large）

## 常见问题

### Q: 如何选择模型规模？
A: 
- 快速实验：nano/small
- 生产环境：medium（推荐）
- 高精度要求：large/xlarge

### Q: 是否必须同时启用高级增强和困难样本挖掘？
A: 不必须。可以单独使用任意一个，或两者结合使用。推荐结合使用以获得最佳效果。

### Q: 训练多久合适？
A: 
- 快速验证：100-500 epochs
- 正式训练：1000-3000 epochs
- 高精度训练：5000+ epochs

### Q: 如何调整数据增强强度？
A: 可以通过额外参数覆盖默认值，例如：
```bash
python train.py --use_advanced_aug --hsv_h 0.02 --scale 0.8
```

## 输出文件说明

训练完成后，`runs/{project_name}/{task_name}/` 目录包含：

```
weights/
  ├── best.pt          # 最佳模型
  ├── last.pt          # 最后一轮模型
  └── best.onnx        # ONNX导出模型
results.csv            # 训练指标CSV
results.png            # 训练曲线图
confusion_matrix.png   # 混淆矩阵
PR_curve.png          # Precision-Recall曲线
F1_curve.png          # F1分数曲线
labels.jpg            # 标签分布
args.yaml             # 训练参数配置
```

## 总结

本脚本针对宫颈细胞识别任务进行了专门优化，通过 `--model_size`、`--use_advanced_aug` 和 `--use_hard_mining` 三个参数，即可快速启用细胞识别的最佳配置。推荐使用示例中的高精度训练配置以获得最佳效果。
