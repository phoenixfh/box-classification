"""
训练配置类

将命令行参数转换为HuggingFace TrainingArguments
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from transformers import TrainingArguments, SchedulerType


@dataclass
class TrainingConfig:
    """
    训练配置参数
    
    支持标准的命令行参数，转换为HuggingFace格式
    """
    # 数据参数
    data_path: str = None
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    
    # 模型参数
    arch: str = 'resnet18'
    pretrained: bool = False
    img_size: int = 224
    
    # 训练参数
    batch_size: int = 256
    epochs: int = 100
    lr0: float = 1e-3
    lrf: float = 0.1
    wd: float = 0.01
    optimizer: str = 'Adam'
    grad_acc: int = -1
    
    # 分布式参数
    distributed: bool = False
    device: Optional[str] = None  # 指定GPU设备，例如 '0', '1' 或 'cuda:0'
    
    # 回调参数
    early_stopping: int = 10
    scheduler_type: str = 'cosine'
    min_lr: Optional[float] = None
    warmup_epochs: int = 3  # 热身阶段的epoch数
    
    # 保存/恢复参数
    model_path: Optional[str] = None
    models_base_dir: str = 'runs'  # 模型保存基础目录
    load_model: Optional[str] = None  # 加载已有模型继续训练（完整路径）
    auto_resume: bool = True  # 自动加载已存在的best模型继续训练

    
    # MLflow参数（与fastai/train.py保持一致）
    project_name: str = 'ai-classifier'  # MLflow实验名称
    task_name: str = 'Image Classification'  # MLflow运行名称
    mlflow_tracking_uri: Optional[str] = None
    skip_mlflow_model_upload: bool = False
    disable_mlflow: bool = False
    
    # 其他参数
    only_val: bool = False
    
    def to_training_arguments(self) -> TrainingArguments:
        """
        转换为HuggingFace TrainingArguments
        
        包含重要的优化和修复:
        - 多GPU模式下的梯度累积处理
        - 自动混合精度训练
        - 合理的保存策略
        - 学习率调度器配置
        """
        # 处理梯度累积
        # 在多GPU模式下，为避免loss双重缩放，禁用梯度累积
        if self.distributed and self.grad_acc > 1:
            if is_main_process():
                print("⚠️  多GPU模式下禁用梯度累积，避免loss双重缩放")
            grad_acc = 1
        else:
            grad_acc = self.grad_acc if self.grad_acc > 0 else 1
        
        # 确定输出目录 - 与 train.py 中的 model_save_dir 保持一致
        output_dir = str(Path(self.models_base_dir) / self.project_name / self.task_name)
        
        # 映射scheduler_type到HuggingFace的SchedulerType
        scheduler_mapping = {
            'cosine': SchedulerType.COSINE,
            'cosine_restarts': SchedulerType.COSINE_WITH_RESTARTS,
            'linear': SchedulerType.LINEAR,
            'constant': SchedulerType.CONSTANT_WITH_WARMUP,
            'polynomial': SchedulerType.POLYNOMIAL,
        }
        lr_scheduler_type = scheduler_mapping.get(
            self.scheduler_type, 
            SchedulerType.COSINE
        )
        
        # 计算warmup比例（限制在 [0, 1] 范围内）
        warmup_ratio = min(self.warmup_epochs / self.epochs, 1.0) if self.epochs > 0 else 0.0
        
        return TrainingArguments(
            # 基本参数
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # 训练参数
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.lr0,
            weight_decay=self.wd,
            
            # 梯度累积
            gradient_accumulation_steps=grad_acc,
            
            # 评估和保存策略 (新版本使用 eval_strategy)
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Logging
            logging_dir=f'{output_dir}/logs',
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            report_to=["mlflow"] if self.mlflow_tracking_uri else [],
            disable_tqdm=False,  # Keep progress bar on main process
            
            # 性能优化
            fp16=True,  # 自动混合精度
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,  # 禁用未使用参数检测以提升性能
            ddp_broadcast_buffers=True,  # 启用buffer广播，避免NCCL hang
            ddp_timeout=7200,  # NCCL超时时间 (秒), 默认1800可能不够
            
            # 学习率调度器配置
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            
            # 其他
            remove_unused_columns=False,
            push_to_hub=False,
            
            # 分布式训练由 accelerate launch 自动配置
        )
    
    @classmethod
    def from_dict(cls, args_dict):
        """从字典创建配置实例"""
        return cls(**args_dict)
