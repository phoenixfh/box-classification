"""
MLflow 集成工具
"""

import os
import mlflow
from pathlib import Path
from fastai.callback.core import Callback
from fastai.callback.tracker import Recorder

from .data_loading import is_main_process


def setup_mlflow(project_name, task_name, tracking_uri=None):
    """
    设置 MLflow 实验和运行
    
    Args:
        project_name: 项目/实验名称
        task_name: 任务/运行名称  
        tracking_uri: MLflow Tracking URI（可选）
        
    Returns:
        mlflow.ActiveRun: MLflow 运行对象，如果非主进程返回 None
    """
    if not is_main_process():
        return None
    
    if tracking_uri is None:
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://192.168.16.130:5000/')

    # 设置MinIO访问凭据
    os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'mlflow@SN'
    os.environ['AWS_ENDPOINT_URL'] = 'http://192.168.16.130:9000'
    os.environ['AWS_REGION'] = ''
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(project_name)
    
    run = mlflow.start_run(run_name=task_name)
    return run


def upload_figure_to_mlflow(figure, title, run_id):
    """上传图表到 MLflow"""
    try:
        # 检查是否已有active run
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            # 如果已经有对应的active run，直接使用
            mlflow.log_figure(figure, f"evaluation/{title}.png")
        else:
            # 否则启动新的run
            with mlflow.start_run(run_id=run_id):
                mlflow.log_figure(figure, f"evaluation/{title}.png")
        print(f"   ✅ 已上传图片到 MLflow: {title}")
    except Exception as e:
        print(f"   ⚠️  上传图片到 MLflow 失败: {e}")


def upload_metrics_to_mlflow(report, classes, run_id):
    """上传指标到 MLflow"""
    try:
        metrics = {}
        
        # 上报每个类别的指标
        for class_name in classes:
            if class_name in report:
                m = report[class_name]
                metrics[f'evaluation/{class_name}/precision'] = m['precision']
                metrics[f'evaluation/{class_name}/recall'] = m['recall']
                metrics[f'evaluation/{class_name}/f1_score'] = m['f1-score']
        
        # 上报总体指标
        if 'accuracy' in report:
            metrics['evaluation/accuracy'] = report['accuracy']
        metrics['evaluation/macro_avg/precision'] = report['macro avg']['precision']
        metrics['evaluation/macro_avg/recall'] = report['macro avg']['recall']
        metrics['evaluation/macro_avg/f1_score'] = report['macro avg']['f1-score']
        metrics['evaluation/weighted_avg/precision'] = report['weighted avg']['precision']
        metrics['evaluation/weighted_avg/recall'] = report['weighted avg']['recall']
        metrics['evaluation/weighted_avg/f1_score'] = report['weighted avg']['f1-score']
        
        # 检查是否已有active run
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            # 如果已经有对应的active run，直接使用
            mlflow.log_metrics(metrics)
        else:
            # 否则启动新的run
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)
        
        print(f"   ✅ 已上传 {len(metrics)} 个指标到 MLflow")
        
    except Exception as e:
        print(f"   ⚠️  上传指标到 MLflow 失败: {e}")


def upload_artifact_to_mlflow(file_path, run_id):
    """上传文件到 MLflow"""
    try:
        # 检查是否已有active run
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            # 如果已经有对应的active run，直接使用
            mlflow.log_artifact(str(file_path), artifact_path="evaluation")
        else:
            # 否则启动新的run
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(file_path), artifact_path="evaluation")
        print(f"   ✅ 已上传文件到 MLflow: {Path(file_path).name}")
    except Exception as e:
        print(f"   ⚠️  上传文件到 MLflow 失败: {e}")


class MLflowMetricsCallback(Callback):
    """MLflow 指标上报回调"""
    def __init__(self, resume_from_epoch=0):
        self.order = Recorder.order + 1  # 确保在 Recorder 之后执行
        self.resume_from_epoch = resume_from_epoch
        
    def after_epoch(self):
        """每个 epoch 结束后上报指标到 MLflow"""
        if not is_main_process():
            return
        
        # 额外检查 MLflow run 是否存在
        if mlflow.active_run() is None:
            print(f"⚠️  警告: Epoch {self.learn.epoch} - 没有活跃的 MLflow run，跳过指标上报")
            return
        
        # 计算实际的 epoch（考虑 resume）
        actual_epoch = self.learn.epoch + self.resume_from_epoch
        
        try:
            metrics_dict = {}
            
            # 上报当前学习率
            if len(self.learn.recorder.lrs) > 0:
                current_lr = self.learn.recorder.lrs[-1]
                metrics_dict['learning_rate'] = current_lr
            
            # 获取 fastai recorder 中记录的指标
            if len(self.learn.recorder.values) == 0:
                return
                
            last_values = self.learn.recorder.values[-1]
            metric_names = self.learn.recorder.metric_names[1:]  # 跳过 'epoch'

            for i, (name, value) in enumerate(zip(metric_names, last_values)):
                if name == 'time':
                    # 跳过 time 指标
                    continue
                metrics_dict[f'{name}'] = value
            
            # 批量上报所有指标
            if metrics_dict:
                mlflow.log_metrics(metrics_dict, step=actual_epoch)
                
        except Exception as e:
            print(f"⚠️  Epoch {actual_epoch} - 记录指标到 MLflow 失败: {e}")
