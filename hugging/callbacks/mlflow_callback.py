"""
from ._utils import is_main_process, print_main
MLflow集成回调

记录训练过程中的metrics和模型
"""

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import mlflow
import os
from typing import Optional
from hugging.utils import print_main


class MLflowCallback(TrainerCallback):
    """
    MLflow集成回调
    
    功能:
    - 自动记录训练参数
    - 记录训练metrics（使用FastAI命名规范）
    - 可选上传模型到MLflow
    """
    
    # HuggingFace → FastAI 指标名称映射表
    METRIC_NAME_MAPPING = {
        # 验证指标（去除 eval_ 前缀）
        'eval_loss': 'valid_loss',
        'eval_accuracy': 'accuracy',
        'eval_precision': 'precision',
        'eval_recall': 'recall',
        'eval_f1': 'f1_score',
        
        # 训练指标
        'loss': 'train_loss',
        
        # 其他（保持不变）
        'learning_rate': 'learning_rate',
        'epoch': 'epoch',
    }
    
    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        skip_model_upload: bool = False,
        tracking_uri: Optional[str] = None
    ):
        """
        Args:
            project_name: MLflow实验名称（对应fastai的project_name）
            task_name: MLflow运行名称（对应fastai的task_name）
            skip_model_upload: 是否跳过模型上传
            tracking_uri: MLflow Tracking URI（默认从环境变量或使用默认值）
        """
        self.project_name = project_name
        self.task_name = task_name
        self.skip_model_upload = skip_model_upload
        self.tracking_uri = tracking_uri
        self.run = None
        self.started = False
    
    @staticmethod
    def convert_metric_name(name: str) -> str:
        """
        转换 HuggingFace 指标名称为 FastAI 风格
        
        Args:
            name: 原始指标名称（HuggingFace格式）
            
        Returns:
            转换后的指标名称（FastAI格式）
        """
        # 优先精确匹配
        if name in MLflowCallback.METRIC_NAME_MAPPING:
            return MLflowCallback.METRIC_NAME_MAPPING[name]
        
        # 处理未知的 eval_* 指标（自动去除前缀）
        if name.startswith('eval_'):
            return name[5:]  # 去除 'eval_' 前缀
        
        # 保持原名称
        return name
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """训练开始时设置MLflow"""
        if not state.is_local_process_zero:
            return
        
        if self.project_name and not self.started:
            # 设置MLflow tracking URI和MinIO凭据（与原版一致）
            tracking_uri = self.tracking_uri or os.environ.get('MLFLOW_TRACKING_URI', 'http://192.168.16.130:5000/')
            
            # 设置MinIO访问凭据
            os.environ['AWS_ACCESS_KEY_ID'] = os.environ.get('AWS_ACCESS_KEY_ID', 'mlflow')
            os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ.get('AWS_SECRET_ACCESS_KEY', 'mlflow@SN')
            os.environ['AWS_ENDPOINT_URL'] = os.environ.get('AWS_ENDPOINT_URL', 'http://192.168.16.130:9000')
            os.environ['AWS_REGION'] = os.environ.get('AWS_REGION', '')
            os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
            
            mlflow.set_tracking_uri(tracking_uri)
            
            # 设置实验（使用project_name）
            mlflow.set_experiment(self.project_name)
            
            # 开始运行（使用task_name）
            self.run = mlflow.start_run(run_name=self.task_name)
            self.started = True
            
            print_main(f"📊 MLflow实验已启动")
            print_main(f"   Tracking URI: {tracking_uri}")
            print_main(f"   项目名称 (Experiment): {self.project_name}")
            if self.task_name:
                print_main(f"   任务名称 (Run): {self.task_name}")
            print_main(f"   Run ID: {self.run.info.run_id}")
            
            # 记录训练参数
            params = {
                'model_name': model.__class__.__name__ if model else 'Unknown',
                'num_train_epochs': args.num_train_epochs,
                'per_device_train_batch_size': args.per_device_train_batch_size,
                'per_device_eval_batch_size': args.per_device_eval_batch_size,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'fp16': args.fp16,
                'output_dir': args.output_dir,
            }
            
            mlflow.log_params(params)
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        """记录metrics到MLflow（使用FastAI命名规范）"""
        if not state.is_local_process_zero or not self.started:
            return
        
        if logs is not None and self.run:
            # 转换为 FastAI 命名风格
            converted_metrics = {}
            for k, v in logs.items():
                if v is not None:
                    new_name = self.convert_metric_name(k)
                    converted_metrics[new_name] = v
            
            # 记录到MLflow
            if converted_metrics:
                mlflow.log_metrics(converted_metrics, step=state.global_step)
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """训练结束时保存模型"""
        if not state.is_local_process_zero or not self.started:
            return
        
        if self.run:
            # 上传最佳模型
            if not self.skip_model_upload and model is not None:
                try:
                    print_main(f"📤 上传模型到MLflow...")
                    mlflow.pytorch.log_model(model, name="model")
                    print_main(f"✅ 模型已上传")
                except Exception as e:
                    print_main(f"⚠️  模型上传失败: {e}")
            
            # 结束运行
            mlflow.end_run()
            self.started = False
            print_main(f"📊 MLflow运行已结束")
