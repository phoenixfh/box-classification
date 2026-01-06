#!/usr/bin/env python3
"""
HuggingFace图像分类训练脚本

使用方式（命令行参数完全兼容原有脚本）:
    python hugging/train.py \\
        --data_path /path/to/data \\
        --arch resnet18 \\
        --batch_size 256 \\
        --epochs 100 \\
        --lr0 0.01 \\
        --distributed

分布式训练:
    accelerate launch hugging/train.py \\
        --data_path /path/to/data \\
        --arch resnet18 \\
        --distributed
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import Trainer
from accelerate import Accelerator

from hugging.config import TrainingConfig
from hugging.models import create_classification_model
from hugging.data import ImageDataset, ImageCollator
from hugging.callbacks import MLflowCallback, SaveModelCallback
from hugging.optimizers import create_optimizer
from hugging.metrics import compute_metrics
from hugging.utils import print_main, generate_evaluation_reports


class CustomTrainer(Trainer):
    """自定义Trainer，支持训练和验证使用不同的data collator"""
    
    def __init__(self, *args, val_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_collator = val_collator
        self.train_collator = self.data_collator
    
    def evaluate(self, *args, **kwargs):
        """重写evaluate方法，使用验证专用的collator"""
        # 临时切换到验证collator
        if self.val_collator is not None:
            original_collator = self.data_collator
            self.data_collator = self.val_collator
            try:
                metrics = super().evaluate(*args, **kwargs)
            finally:
                # 恢复训练collator
                self.data_collator = original_collator
        else:
            metrics = super().evaluate(*args, **kwargs)
        return metrics
    
    def predict(self, *args, **kwargs):
        """重写predict方法，使用验证专用的collator"""
        # 临时切换到验证collator
        if self.val_collator is not None:
            original_collator = self.data_collator
            self.data_collator = self.val_collator
            try:
                result = super().predict(*args, **kwargs)
            finally:
                # 恢复训练collator
                self.data_collator = original_collator
        else:
            result = super().predict(*args, **kwargs)
        return result


def parse_args():
    """
    解析命令行参数
    
    保持与原有脚本完全一致的参数名称
    """
    parser = argparse.ArgumentParser(
        description='HuggingFace图像分类训练脚本'
    )
    
    # === 数据参数 ===
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='数据集路径（包含train/val子目录）'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=None,
        help='训练集大小限制'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=None,
        help='验证集大小限制'
    )
    
    # === 模型参数 ===
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet18',
        help='模型架构 (resnet18, resnet50, vit_base, etc.)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='使用预训练权重'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='图像大小'
    )
    
    # === 训练参数 ===
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='每个设备的batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='初始学习率'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.1,
        help='最终学习率比例 (final_lr = lr0 * lrf)'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=0.01,
        help='权重衰减 (weight decay)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        choices=['SGD', 'Adam', 'AdamW', 'RMSprop'],
        help='优化器类型'
    )
    parser.add_argument(
        '--grad_acc',
        type=int,
        default=-1,
        help='梯度累积步数 (-1表示不使用)'
    )
    
    # === 分布式参数 ===
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='启用分布式训练（需要使用accelerate launch）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='指定GPU设备，例如 "0", "1" 或 "cuda:0", None=使用默认'
    )
    
    # === 回调参数 ===
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=100,
        help='Early stopping的patience（默认100）'
    )
    parser.add_argument(
        '--scheduler_type',
        type=str,
        default='cosine',
        choices=['cosine', 'cosine_restarts', 'linear', 'constant', 'polynomial'],
        help='学习率调度器类型'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=3,
        help='热身阶段的epoch数（默认3）'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=None,
        help='最小学习率（默认为lr0的1%）'
    )
    
    # === 保存/恢复参数 ===
    parser.add_argument(
        '--model_path',
        type=str,
        default='last',
        help='保存模型的文件名（不含路径，默认: last）'
    )
    parser.add_argument(
        '--models_base_dir',
        type=str,
        default='runs',
        help='统一的模型保存基础目录，实际保存路径为: models_base_dir/project_name/task_name/'
    )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='加载已有的模型继续训练（完整路径）'
    )
    
    parser.add_argument(
        '--project_name',
        type=str,
        default='ai-classifier',
        help='MLflow实验名称（默认: ai-classifier）'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default='Image Classification',
        help='MLflow运行名称（默认: Image Classification）'
    )
    parser.add_argument(
        '--mlflow_tracking_uri',
        type=str,
        default=None,
        help='MLflow Tracking URI（默认: 从环境变量或使用默认值）'
    )
    parser.add_argument(
        '--skip_mlflow_model_upload',
        action='store_true',
        help='跳过模型上传到MLflow'
    )
    parser.add_argument(
        '--disable_mlflow',
        action='store_true',
        help='禁用MLflow（默认启用）'
    )
    
    # === 其他参数 ===
    parser.add_argument(
        '--only_val',
        action='store_true',
        help='仅执行验证（不训练）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    

    # 检查是否为主进程（用于输出控制）
    is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    print_main("=" * 80)
    print_main("🚀 HuggingFace图像分类训练")
    print_main("=" * 80)
    
    if is_main:
        # 显示模型保存路径信息
        print_main(f"\n{'='*80}")
        print_main(f"模型保存配置:")
        print_main(f"  基础目录: {args.models_base_dir}")
        print_main(f"  项目名称: {args.project_name}")
        print_main(f"  任务名称: {args.task_name}")
        print_main(f"  实际路径: {Path(args.models_base_dir).absolute() / args.project_name / args.task_name}")
        print_main(f"{'='*80}\n")
    
    # 0. 处理GPU设备选择
    if args.device is not None:
        import os
        device_id = args.device.split(':')[-1] if ':' in args.device else args.device
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    print_main(f"\n🎮 使用指定GPU: {args.device}")
    
    # 1. 加载数据集
    print_main(f"\n📁 加载数据集...")

    dataset_info = ImageDataset.from_directory(
        data_path=args.data_path,
        train_size=args.train_size,
        val_size=args.val_size,
        img_size=args.img_size
    )
    
    train_dataset = dataset_info['train']
    val_dataset = dataset_info['val']
    num_classes = dataset_info['num_classes']
    
    # 2. 创建模型
    print_main(f"\n🏗️  创建模型: {args.arch}")
    print_main(f"   类别数: {num_classes}")
    print_main(f"   预训练: {'是' if args.pretrained else '否'}")
    
    model = create_classification_model(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained
    )
    
    # 2.5 处理模型加载和恢复
    resume_from_epoch = 0
    resume_best_metric = None
    optimizer_state_to_load = None
    scheduler_state_to_load = None
    
    # 构建统一的模型保存目录: models_base_dir/project_name/task_name/
    model_save_dir = Path(args.models_base_dir) / args.project_name / args.task_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    print_main(f"\n📁 模型保存目录: {model_save_dir.absolute()}")
    
    # 自动恢复逻辑：如果没有明确指定load_model，且启用了auto_resume，尝试自动加载
    load_model_path = args.load_model
    if load_model_path is None:
        auto_load_path = model_save_dir / 'best.pth'
        if auto_load_path.exists():
            print_main(f"\n🔍 发现已存在的模型: {auto_load_path}")
            print_main(f"   自动加载以继续训练...")
            load_model_path = str(auto_load_path)
    
    # 加载checkpoint
    if load_model_path is not None:
        print_main(f"\n{'='*80}")
        print_main(f"📦 从 checkpoint 加载模型")
        print_main(f"{'='*80}")
        print_main(f"模型路径: {load_model_path}")
        
        # 加载状态字典
        checkpoint = torch.load(load_model_path, map_location='cpu')
        
        print_main(f"\n📋 Checkpoint 信息:")
        
        # 检查是否包含epoch信息
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            resume_from_epoch = checkpoint['epoch'] + 1  # +1因为保存的是完成的epoch
            print_main(f"  - 已完成的 epoch: {checkpoint['epoch']}")
            print_main(f"  - 下次训练起始 epoch: {resume_from_epoch}")
        else:
            print_main(f"  - Epoch 信息: ⚠️  未找到")
        
        # 检查loss信息
        if isinstance(checkpoint, dict) and 'loss' in checkpoint:
            resume_best_metric = checkpoint['loss']
            print_main(f"  - 当前 valid_loss: {resume_best_metric:.6f}")
        else:
            print_main(f"  - 最佳指标: ⚠️  未找到")
        
        # 检查其他信息
        if isinstance(checkpoint, dict):
            if 'img_size' in checkpoint:
                print_main(f"  - 图像尺寸: {checkpoint['img_size']}")
            if 'arch' in checkpoint:
                print_main(f"  - 模型架构: {checkpoint['arch']}")
            if 'opt' in checkpoint:
                print_main(f"  - 优化器状态: ✅ 已保存")
                optimizer_state_to_load = checkpoint['opt']
            else:
                print_main(f"  - 优化器状态: ⚠️  未找到")
            if 'scheduler' in checkpoint:
                print_main(f"  - 调度器状态: ✅ 已保存")
                scheduler_state_to_load = checkpoint['scheduler']
            else:
                print_main(f"  - 调度器状态: ⚠️  未找到")
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            model_state = checkpoint  # 纯模型权重
        
        # 处理DDP包装的权重名称
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        
        print_main(f"\n✅ 模型权重加载成功")
        if resume_from_epoch > 0:
            print_main(f"📊 将从 epoch {resume_from_epoch} 继续训练")
    
    # 3. 创建训练配置
    print_main(f"\n⚙️  配置训练参数...")
    print_main(f"   传入batch_size: {args.batch_size}")

    config = TrainingConfig(**vars(args))
    training_args = config.to_training_arguments()

    print_main(f"   实际per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print_main(f"   实际per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
    
    # 显示学习率调度器配置
    print_main(f"\n📊 学习率调度器配置:")
    print_main(f"   类型: {args.scheduler_type}")
    print_main(f"   初始学习率 (lr0): {args.lr0}")
    print_main(f"   最终学习率比例 (lrf): {args.lrf}")
    print_main(f"   最终学习率: {args.lr0 * args.lrf}")
    print_main(f"   热身epochs: {args.warmup_epochs}")
    print_main(f"   热身比例: {args.warmup_epochs / args.epochs if args.epochs > 0 else 0:.2%}")
    if args.min_lr:
        print_main(f"   最小学习率: {args.min_lr}")
    else:
        print_main(f"   最小学习率: {args.lr0 * 0.01} (默认为lr0的1%)")
    
    # 4. 创建优化器
    optimizer = create_optimizer(
        optimizer_name=args.optimizer,
        model=model,
        lr=args.lr0,
        weight_decay=args.wd
    )
    
    # 4.5 加载优化器状态（如果有）
    if optimizer_state_to_load is not None:
        try:
            optimizer.load_state_dict(optimizer_state_to_load)
            print_main(f"\n💾 优化器状态已恢复")
            # 显示恢复的学习率
            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                restored_lr = optimizer.param_groups[0].get('lr', 'N/A')
                print_main(f"   恢复的学习率: {restored_lr}")
        except Exception as e:
            print_main(f"\n⚠️  优化器状态加载失败: {e}")
            print_main(f"   将使用新的优化器状态")

    ## 输出优化器中的 lr 参数
    if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
        initial_lr = optimizer.param_groups[0].get('lr', 'N/A')
        print_main(f"\n🔧 优化器当前学习率: {initial_lr}")
    
    # 4.6 创建学习率调度器
    from transformers import get_scheduler
    
    # 计算训练步数
    steps_per_epoch = len(train_dataset) // (args.batch_size * training_args.gradient_accumulation_steps)
    if args.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            steps_per_epoch = steps_per_epoch // dist.get_world_size()
    
    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = steps_per_epoch * args.warmup_epochs
    
    print_main(f"\n📊 训练步数计算:")
    print_main(f"   每个epoch步数: {steps_per_epoch}")
    print_main(f"   总训练步数: {num_training_steps}")
    print_main(f"   热身步数: {num_warmup_steps}")
    
    # 映射scheduler_type到HuggingFace的SchedulerType
    from transformers import SchedulerType
    scheduler_mapping = {
        'cosine': SchedulerType.COSINE,
        'cosine_restarts': SchedulerType.COSINE_WITH_RESTARTS,
        'linear': SchedulerType.LINEAR,
        'constant': SchedulerType.CONSTANT_WITH_WARMUP,
        'polynomial': SchedulerType.POLYNOMIAL,
    }
    lr_scheduler_type = scheduler_mapping.get(args.scheduler_type, SchedulerType.COSINE)
    
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 4.7 加载调度器状态（如果有）
    if scheduler_state_to_load is not None:
        try:
            lr_scheduler.load_state_dict(scheduler_state_to_load)
            print_main(f"\n💾 学习率调度器状态已恢复")
        except Exception as e:
            print_main(f"\n⚠️  调度器状态加载失败: {e}")
            print_main(f"   将使用新的调度器状态")
      
    # 5. 创建Data Collator
    train_collator = ImageCollator(img_size=args.img_size, is_training=True)
    val_collator = ImageCollator(img_size=args.img_size, is_training=False)
    
    # 6. 创建Callbacks
    callbacks = []
    
    # MLflow集成（默认启用，参数名与fastai/train.py一致）
    if not args.disable_mlflow:
        task_name = args.task_name
        if not task_name:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            task_name = f"{args.arch}_{timestamp}"
        
        callbacks.append(
            MLflowCallback(
                project_name=args.project_name,
                task_name=task_name,
                skip_model_upload=args.skip_mlflow_model_upload,
                tracking_uri=args.mlflow_tracking_uri
            )
        )

    print_main(f"   项目 (Experiment): {args.project_name}")
    print_main(f"   任务 (Run): {task_name}")
    
    # 保存模型callback（保存完整的checkpoint信息，支持early stopping）
    save_model_callback = SaveModelCallback(
        img_size=args.img_size,
        arch=args.arch,
        resume_from_epoch=resume_from_epoch,
        patience=args.early_stopping,
        monitor='eval_loss',
        mode='min'
    )
    callbacks.append(save_model_callback)
    print_main(f"💾 已启用模型保存 (early_stopping patience={args.early_stopping})")
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,  # 训练用collator
        val_collator=val_collator,  # 验证用collator
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler),  # (optimizer, lr_scheduler)
    )
    
    # 设置trainer引用给save_model_callback，以便访问optimizer
    save_model_callback.trainer = trainer
    
    # 8. 训练或验证
    if args.only_val:
        print_main(f"\n📊 验证模式...")
        metrics = trainer.evaluate()
        print_main(f"\n验证结果:")
        for key, value in metrics.items():
            print_main(f"  {key}: {value:.4f}")
    else:
        print_main(f"\n🏋️  开始训练...")
        trainer.train()
        
    # 训练结束后在验证集上评估
    print_main(f"\n📊 最终验证...")
    metrics = trainer.evaluate()
    print_main(f"\n最终验证结果:")
    for key, value in metrics.items():
        print_main(f"  {key}: {value:.4f}")
    
    # 9. 生成详细评估报告（仅非验证模式且主进程）
    if not args.only_val and trainer.args.local_rank in [-1, 0]:
        print_main(f"\n📊 生成详细评估报告...")
        
        # 获取验证集的预测结果
        predictions = trainer.predict(val_dataset)
        pred_logits = predictions.predictions
        pred_labels = predictions.label_ids
        
        # 获取类别列表（从dataset_info中获取）
        classes = dataset_info['labels']
        
        # 获取MLflow run ID（如果启用）
        mlflow_run_id = None
        if not args.disable_mlflow:
            for callback in trainer.callback_handler.callbacks:
                if isinstance(callback, MLflowCallback) and callback.run:
                    mlflow_run_id = callback.run.info.run_id
                    break
        
        # 生成评估报告
        generate_evaluation_reports(
            predictions=pred_logits,
            targets=pred_labels,
            classes=classes,
            output_dir=Path(training_args.output_dir),
            mlflow_run_id=mlflow_run_id
        )
    
    print_main(f"\n✅ 完成！")

    if not args.only_val:
        print_main(f"   模型保存在: {training_args.output_dir}")


if __name__ == '__main__':
    main()
