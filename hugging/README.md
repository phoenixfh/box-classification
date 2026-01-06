# HuggingFace图像分类训练

基于HuggingFace Transformers和Accelerate的图像分类训练脚本，完全兼容FastAI版本的命令行参数。

## 主要特性

### 1. **模型加载和自动恢复**
- ✅ 支持从checkpoint恢复训练（`--load_model`）
- ✅ 自动恢复功能（`--auto_resume`，默认启用）
- ✅ 完整保存：模型权重、优化器状态、epoch、loss等
- ✅ 正确恢复学习率调度（基于resume_from_epoch）

### 2. **命令行参数**
完全兼容FastAI版本，主要参数包括：

```bash
# 数据参数
--data_path         # 数据集路径
--train_size        # 训练集大小限制
--val_size          # 验证集大小限制

# 模型参数
--arch              # 模型架构（resnet18, resnet34等）
--pretrained        # 使用预训练权重
--img_size          # 图像尺寸

# 训练参数
--batch_size        # 批大小
--epochs            # 训练轮数
--lr0               # 初始学习率
--lrf               # 最终学习率比例
--wd                # 权重衰减
--optimizer         # 优化器类型（SGD, Adam, AdamW）

# 模型加载和恢复
--load_model        # 加载已有模型继续训练（完整路径）
--auto_resume       # 自动加载best.pth继续训练（默认启用）
--no-auto-resume    # 禁用自动恢复

# MLflow参数（与FastAI保持一致）
--project_name      # MLflow实验名称
--task_name         # MLflow运行名称
--mlflow_tracking_uri  # MLflow服务地址
--disable_mlflow    # 禁用MLflow
--skip_mlflow_model_upload  # 跳过模型上传

# 设备参数
--device            # 指定GPU（如 "0,1,2,3"）
--distributed       # 启用分布式训练
```

### 3. **使用示例**

#### 单GPU训练
```bash
python hugging/train.py \
    --data_path /mnt/ssd/ai-classify-all-data \
    --arch resnet34 \
    --batch_size 256 \
    --epochs 100 \
    --lr0 0.01 \
    --device 0
```

#### 多GPU训练
```bash
accelerate launch hugging/train.py \
    --data_path /mnt/ssd/ai-classify-all-data \
    --arch resnet34 \
    --batch_size 256 \
    --epochs 100 \
    --lr0 0.01 \
    --distributed
```

#### 从checkpoint恢复训练
```bash
# 方式1：自动恢复（默认行为，会加载./models/best.pth）
python hugging/train.py \
    --data_path /mnt/ssd/ai-classify-all-data \
    --arch resnet34

# 方式2：指定checkpoint路径
python hugging/train.py \
    --data_path /mnt/ssd/ai-classify-all-data \
    --arch resnet34 \
    --load_model /path/to/checkpoint.pth

# 方式3：禁用自动恢复
python hugging/train.py \
    --data_path /mnt/ssd/ai-classify-all-data \
    --arch resnet34 \
    --no-auto-resume
```

### 4. **Checkpoint格式**

保存的checkpoint包含以下信息：
```python
{
    'model': model_state_dict,      # 模型权重
    'opt': optimizer_state_dict,    # 优化器状态
    'epoch': int,                   # 已完成的epoch数
    'loss': float,                  # 当前验证loss
    'img_size': int,                # 图像尺寸
    'arch': str,                    # 模型架构
}
```

### 5. **与FastAI版本的差异**

#### 相同点：
- ✅ 完全相同的命令行参数
- ✅ 相同的MLflow集成方式（**指标命名已统一到FastAI风格**）
- ✅ 相同的学习率调度策略
- ✅ 相同的checkpoint格式
- ✅ 相同的自动恢复逻辑

#### 不同点：
- 🔄 使用HuggingFace Trainer替代FastAI Learner
- 🔄 使用Accelerate进行分布式训练
- 🔄 更好的多GPU支持和验证loss计算
- 🔄 没有FastAI依赖，纯HuggingFace生态

### 6. **MLflow指标命名**

为了与FastAI训练脚本保持一致，HuggingFace版本会自动将指标名称转换为FastAI风格：

| HuggingFace原始名称 | MLflow中的名称 (FastAI风格) |
|-------------------|---------------------------|
| `eval_loss` | `valid_loss` |
| `eval_accuracy` | `accuracy` |
| `eval_precision` | `precision` |
| `eval_recall` | `recall` |
| `eval_f1` | `f1_score` |
| `loss` | `train_loss` |
| `learning_rate` | `learning_rate` |

**注意：** 这仅影响上报到MLflow的指标名称，训练日志中仍使用HuggingFace原始名称。

### 7. **关键修复**

针对多GPU训练的验证loss问题，实现了：
1. ✅ 正确的分布式验证集采样（避免数据重复）
2. ✅ 准确的loss聚合（跨GPU平均）
3. ✅ 诊断工具（ValidationDiagnosticCallback）

### 8. **模型保存路径**

- 默认保存路径：`./models/`
- Best模型：`./models/best.pth`
- HuggingFace checkpoints：`./models/checkpoint-xxx/`

可通过 `--model_path` 自定义保存路径。

## 架构组件

```
hugging/
├── train.py                    # 主训练脚本
├── config/
│   └── training_args.py       # 训练配置类
├── models/
│   ├── __init__.py
│   └── model_builder.py       # 模型构建器
├── data/
│   ├── __init__.py
│   ├── dataset.py             # 数据集类
│   └── collator.py            # 数据整理器
├── callbacks/
│   ├── __init__.py
│   ├── yolov11_lr_scheduler.py  # 学习率调度器
│   ├── mlflow_callback.py       # MLflow集成
│   ├── save_model.py            # 模型保存回调
│   └── validation_diagnostic.py # 验证诊断工具
├── optimizers/
│   └── optimizer_factory.py   # 优化器工厂
└── metrics/
    └── classification_metrics.py  # 分类指标
```

## 常见问题

### Q: 如何禁用自动恢复？
A: 使用 `--no-auto-resume` 参数

### Q: 如何指定使用哪些GPU？
A: 使用 `--device "0,1,2,3"` 参数

### Q: 验证loss与训练loss差距很大？
A: 这可能是正常现象（过拟合），可以：
- 增加数据增强
- 增加weight decay（`--wd 0.001`）
- 使用early stopping（`--early_stopping 10`）

### Q: 如何查看训练进度？
A: 通过MLflow UI查看（默认 http://localhost:5000）

## 贡献

欢迎提交问题和改进建议！
