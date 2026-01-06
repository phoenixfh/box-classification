"""
保存模型回调

保存完整的checkpoint信息，包括epoch、loss、优化器状态等
支持early stopping
"""

import torch
from pathlib import Path
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os
from hugging.utils import print_main

class SaveModelCallback(TrainerCallback):
    """
    保存完整的checkpoint信息
    
    保存内容:
    - model: 模型权重
    - opt: 优化器状态
    - epoch: 当前epoch
    - loss: 当前loss
    - img_size: 图像尺寸
    - arch: 模型架构
    
    支持early stopping功能
    """
    
    def __init__(
        self,
        img_size: int,
        arch: str,
        resume_from_epoch: int = 0,
        patience: int = 100,
        monitor: str = 'eval_loss',
        mode: str = 'min'
    ):
        """
        Args:
            img_size: 图像尺寸
            arch: 模型架构名称
            resume_from_epoch: 恢复训练的起始epoch
            patience: 早停patience（验证loss不改善的epoch数）
            monitor: 监控的指标名称
            mode: 'min' 或 'max'，指标是越小越好还是越大越好
        """
        self.img_size = img_size
        self.arch = arch
        self.resume_from_epoch = resume_from_epoch
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        
        # Early stopping state
        self.best_metric = None
        self.wait = 0
        self.stopped_epoch = 0
        
        # Store trainer reference
        self.trainer = None
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Store trainer reference at training begin"""
        # Get trainer from kwargs (Trainer passes itself to callbacks)
        if 'model' in kwargs:
            # Find the trainer instance - it should be accessible through the call stack
            # We'll store it when we get it in on_evaluate
            pass
    
    def _is_better(self, current, best):
        """判断当前指标是否更好"""
        if best is None:
            return True
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        model=None,
        **kwargs
    ):
        """
        评估后检查是否需要保存最优模型和early stopping
        """
        # 只在主进程执行
        if not state.is_local_process_zero:
            return control
        
        if metrics is None or self.monitor not in metrics:
            return control
        
        current_metric = metrics[self.monitor]
        
        # Model is passed by Trainer automatically
        # Try to get optimizer and lr_scheduler from kwargs - Trainer might pass it
        optimizer = None
        lr_scheduler = None
        
        # Check if we can get the trainer instance
        # The trainer instance should have the optimizer and lr_scheduler
        if self.trainer is not None:
            optimizer = getattr(self.trainer, 'optimizer', None)
            lr_scheduler = getattr(self.trainer, 'lr_scheduler', None)
        
        if model is None:
            return control
        
        # 检查是否是最优模型
        if self._is_better(current_metric, self.best_metric):
            # 保存最优模型
            self.best_metric = current_metric
            self.wait = 0
            
            # 保存best.pth
            output_dir = Path(args.output_dir)
            best_path = output_dir / 'best.pth'
            
            # 获取实际的epoch
            actual_epoch = int(state.epoch) + self.resume_from_epoch - 1 if state.epoch else 0
            
            # 获取模型状态（处理DDP包装）
            if hasattr(model, 'module'):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            
            # 构建完整的checkpoint
            checkpoint = {
                'model': model_state,
                'epoch': actual_epoch,
                'img_size': self.img_size,
                'arch': self.arch,
                'loss': current_metric,
            }
            
            # 添加优化器状态
            if optimizer is not None:
                checkpoint['opt'] = optimizer.state_dict()
            
            # 添加学习率调度器状态
            if lr_scheduler is not None:
                checkpoint['scheduler'] = lr_scheduler.state_dict()
            
            # 保存
            torch.save(checkpoint, best_path)
            
            print_main(f"\n💾 保存最优模型到: {best_path}")
            print_main(f"   - Epoch: {actual_epoch}")
            print_main(f"   - {self.monitor}: {current_metric:.6f} (improved)")
            print_main(f"   - lr: {optimizer.param_groups[0]['lr'] if optimizer else 'N/A'}")
            print_main(f"   - 包含优化器状态: {'✅' if optimizer else '❌'}")
            print_main(f"   - 包含调度器状态: {'✅' if lr_scheduler else '❌'}")
        else:
            # 没有改善
            self.wait += 1
            print_main(f"\n⏳ {self.monitor}: {current_metric:.6f} (no improvement, patience: {self.wait}/{self.patience})")
            
            # 检查是否需要early stopping
            if self.wait >= self.patience:
                self.stopped_epoch = state.epoch
                control.should_training_stop = True
                print_main(f"\n🛑 Early stopping triggered at epoch {int(state.epoch)}")
                print_main(f"   - Best {self.monitor}: {self.best_metric:.6f}")
                print_main(f"   - lr: {optimizer.param_groups[0]['lr'] if optimizer else 'N/A'}")
                print_main(f"   - No improvement for {self.patience} evaluations")
        
        return control
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """训练结束时的总结"""
        if not state.is_local_process_zero:
            return
        
        if self.stopped_epoch > 0:
            print_main(f"\n✅ 训练因early stopping结束于epoch {int(self.stopped_epoch)}")
        
        if self.best_metric is not None:
            print_main(f"\n🏆 最优 {self.monitor}: {self.best_metric:.6f}")
