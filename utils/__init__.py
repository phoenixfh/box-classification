"""
FastAI 训练工具模块

提供训练、评估所需的各种工具类和函数。
"""

# 数据加载工具（不依赖 fastai）
from .data_loading import is_main_process

# 延迟导入 fastai 依赖的模块，避免 YOLO 使用时出错
__all__ = [
    # 数据加载
    'is_main_process',
]

# 只在需要时才导入 fastai 相关模块
def __getattr__(name):
    """延迟导入 fastai 依赖的模块"""
    if name == 'SegmentationDataset' or name == 'get_segmentation_dls':
        from .segmentation import SegmentationDataset, get_segmentation_dls
        return SegmentationDataset if name == 'SegmentationDataset' else get_segmentation_dls
    
    elif name == 'dice_score' or name == 'DiceMetric':
        from .metrics import dice_score, DiceMetric
        return dice_score if name == 'dice_score' else DiceMetric
    
    elif name == 'dice_loss' or name == 'CombinedLoss':
        from .losses import dice_loss, CombinedLoss
        return dice_loss if name == 'dice_loss' else CombinedLoss
    
    elif name in ['YOLOv11LRScheduler', 'LoadOptimizerStateCallback', 'ResumeEpochCallback',
                  'EarlyStoppingWithEvalCallback', 'SaveModelWithEpochCallback',
                  'DistributedValidationDiagnosticCallback']:
        from .callbacks import (
            YOLOv11LRScheduler, LoadOptimizerStateCallback, ResumeEpochCallback,
            EarlyStoppingWithEvalCallback, SaveModelWithEpochCallback,
            DistributedValidationDiagnosticCallback
        )
        return locals()[name]
    
    elif name in ['setup_mlflow', 'MLflowMetricsCallback', 'upload_figure_to_mlflow',
                  'upload_metrics_to_mlflow', 'upload_artifact_to_mlflow']:
        from .mlflow_utils import (
            setup_mlflow, MLflowMetricsCallback, upload_figure_to_mlflow,
            upload_metrics_to_mlflow, upload_artifact_to_mlflow
        )
        return locals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
