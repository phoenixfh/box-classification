"""
MLflow调优实验管理

支持嵌套实验、最佳模型选择、参数重要性分析等
"""

import mlflow
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime


class TuningExperimentManager:
    """MLflow调优实验管理器"""
    
    def __init__(self, project_name: str, task_name: str, config: Dict[str, Any]):
        """
        Args:
            project_name: MLflow实验名称
            task_name: 调优任务名称
            config: 调优配置
        """
        self.project_name = project_name
        self.task_name = task_name
        self.config = config
        self.parent_run = None
        self.trial_runs = []
        
    def start_parent_run(self) -> mlflow.ActiveRun:
        """启动父级MLflow run"""
        import os
        
        mlflow.set_experiment(self.project_name)
        self.parent_run = mlflow.start_run(run_name=self.task_name)
        
        # 记录调优配置
        mlflow.log_params({
            'tuning_strategy': self.config.get('strategy', 'grid'),
            'n_trials': self.config.get('n_trials', 'all'),
            'metric': self.config.get('metric', 'valid_loss'),
            'mode': self.config.get('mode', 'minimize'),
        })
        
        # 记录搜索空间（只使用参数，避免 S3 上传）
        search_space = self.config.get('search_space', {})
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'unknown')
                mlflow.log_param(f'search_space/{param_name}/type', param_type)
                if 'values' in param_config:
                    mlflow.log_param(f'search_space/{param_name}/values', str(param_config['values'])[:250])
                elif 'min' in param_config and 'max' in param_config:
                    mlflow.log_param(f'search_space/{param_name}/min', str(param_config['min']))
                    mlflow.log_param(f'search_space/{param_name}/max', str(param_config['max']))
        
        # 记录固定参数
        base_args = self.config.get('base_args', {})
        for key, value in base_args.items():
            if not key.startswith('_') and key not in ['mlflow_uri']:  # 跳过内部参数
                try:
                    mlflow.log_param(f'base/{key}', str(value)[:250])  # MLflow 限制参数长度
                except Exception:
                    pass  # 忽略参数记录错误
        
        print(f"✅ 启动调优实验: {self.task_name}")
        print(f"   Parent Run ID: {self.parent_run.info.run_id}")
        
        return self.parent_run
    
    def start_trial_run(self, trial_idx: int, trial_params: Dict[str, Any]) -> mlflow.ActiveRun:
        """启动子trial run"""
        # 生成trial名称
        param_summary = self._format_param_summary(trial_params)
        trial_name = f"trial_{trial_idx:03d}_{param_summary}"
        
        # 启动嵌套run
        trial_run = mlflow.start_run(
            run_name=trial_name,
            nested=True,
            parent_run_id=self.parent_run.info.run_id
        )
        
        # 记录trial参数
        for key, value in trial_params.items():
            if not key.startswith('_'):  # 跳过内部字段
                mlflow.log_param(key, value)
        
        mlflow.log_param('trial_idx', trial_idx)
        
        self.trial_runs.append({
            'run_id': trial_run.info.run_id,
            'trial_idx': trial_idx,
            'params': trial_params.copy()
        })
        
        return trial_run
    
    def end_trial_run(self):
        """结束当前trial run"""
        mlflow.end_run()
    
    def end_parent_run(self):
        """结束父级run"""
        if self.parent_run:
            mlflow.end_run()
            print(f"✅ 调优实验完成")
    
    def _format_param_summary(self, params: Dict[str, Any]) -> str:
        """格式化参数摘要（用于run名称）"""
        summary_parts = []
        
        # 自定义显示规则（更简洁的格式）
        # 支持 FastAI 和 YOLO 参数
        param_mappings = {
            # FastAI 参数
            'lr': ('lr', lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)),
            'batch_size': ('bs', lambda v: str(v)),
            'img_size': ('img', lambda v: str(v)),
            'arch': ('', lambda v: str(v)),  # 架构直接显示
            'wd': ('wd', lambda v: f"{v:.1e}" if isinstance(v, float) else str(v)),
            'scale': ('sc', lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)),
            
            # YOLO 参数
            'lr0': ('lr', lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)),
            'batch': ('bs', lambda v: str(v)),
            'imgsz': ('img', lambda v: str(v)),
            'model': ('', lambda v: str(v).replace('.pt', '')),  # 模型名称，去掉 .pt
            'optimizer': ('opt', lambda v: str(v)),
            'weight_decay': ('wd', lambda v: f"{v:.1e}" if isinstance(v, float) else str(v)),
            'degrees': ('deg', lambda v: f"{v:.0f}" if isinstance(v, float) else str(v)),
            'mosaic': ('mos', lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)),
        }
        
        # 按顺序处理参数
        for key, (prefix, formatter) in param_mappings.items():
            if key in params:
                value = params[key]
                formatted = formatter(value)
                if prefix:
                    summary_parts.append(f"{prefix}{formatted}")
                else:
                    summary_parts.append(formatted)
        
        # 最多显示5个参数避免名称过长
        return '_'.join(summary_parts[:5])


def select_best_run(parent_run_id: str, metric: str = 'valid_loss', 
                    mode: str = 'minimize') -> Optional[Dict[str, Any]]:
    """
    从调优实验中选择最佳run
    
    Args:
        parent_run_id: 父级run ID
        metric: 优化目标指标
        mode: 'minimize' 或 'maximize'
        
    Returns:
        最佳run信息字典，包含 run_id, params, metrics
    """
    # 获取所有子runs
    client = mlflow.tracking.MlflowClient()
    
    # 搜索嵌套runs
    runs = client.search_runs(
        experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=[f"metrics.{metric} {'ASC' if mode == 'minimize' else 'DESC'}"]
    )
    
    if not runs:
        print("⚠️ 没有找到子runs")
        return None
    
    best_run = runs[0]
    
    # 提取参数和指标
    best_info = {
        'run_id': best_run.info.run_id,
        'params': best_run.data.params,
        'metrics': best_run.data.metrics,
        'metric_value': best_run.data.metrics.get(metric)
    }
    
    # 标记最佳run
    client.set_tag(best_run.info.run_id, 'best_run', 'true')
    
    print(f"\n🏆 最佳模型:")
    print(f"   Run ID: {best_info['run_id']}")
    if best_info['metric_value'] is not None:
        print(f"   {metric}: {best_info['metric_value']:.6f}")
    else:
        print(f"   {metric}: N/A (未记录)")
    print(f"   参数:")
    for key, value in best_info['params'].items():
        if not key.startswith('base_') and not key.startswith('search_space/'):
            print(f"     - {key}: {value}")
    
    return best_info


def analyze_parameter_importance(parent_run_id: str, metric: str = 'valid_loss',
                                  output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    分析参数重要性
    
    Args:
        parent_run_id: 父级run ID
        metric: 目标指标
        output_dir: 输出目录（可选）
        
    Returns:
        参数重要性DataFrame
    """
    client = mlflow.tracking.MlflowClient()
    
    # 获取所有子runs
    runs = client.search_runs(
        experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )
    
    if len(runs) < 10:
        print(f"⚠️ 试验数量太少（{len(runs)}），建议至少10个试验才能进行参数重要性分析")
        return pd.DataFrame()
    
    # 收集数据
    data = []
    for run in runs:
        row = {**run.data.params, metric: run.data.metrics.get(metric)}
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 计算相关性（简单方法）
    param_cols = [col for col in df.columns if col != metric]
    importance = {}
    
    for param in param_cols:
        if param.startswith('base_'):
            continue
        
        # 尝试转换为数值
        try:
            param_values = pd.to_numeric(df[param], errors='coerce')
            metric_values = pd.to_numeric(df[metric], errors='coerce')
            
            # 计算相关系数
            corr = param_values.corr(metric_values)
            importance[param] = abs(corr) if not np.isnan(corr) else 0
        except:
            # 分类参数，计算方差分析
            groups = df.groupby(param)[metric].apply(list)
            if len(groups) > 1:
                # 简单的方差比
                between_var = groups.apply(np.mean).var()
                within_var = groups.apply(np.var).mean()
                importance[param] = between_var / (within_var + 1e-10)
            else:
                importance[param] = 0
    
    # 排序
    importance_df = pd.DataFrame([
        {'parameter': k, 'importance': v}
        for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
    ])
    
    print(f"\n📊 参数重要性分析 (基于{len(runs)}个试验):")
    for _, row in importance_df.iterrows():
        print(f"   {row['parameter']}: {row['importance']:.4f}")
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_dir / 'parameter_importance.csv', index=False)
        print(f"   保存到: {output_dir / 'parameter_importance.csv'}")
    
    return importance_df


class TuningCheckpoint:
    """调优进度跟踪和恢复"""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, state: Dict[str, Any]):
        """保存checkpoint"""
        state['timestamp'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> Optional[Dict[str, Any]]:
        """加载checkpoint"""
        if not self.checkpoint_path.exists():
            return None
        
        with open(self.checkpoint_path, 'r') as f:
            return json.load(f)
    
    def exists(self) -> bool:
        """检查checkpoint是否存在"""
        return self.checkpoint_path.exists()
    
    def clear(self):
        """清除checkpoint"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


def log_tuning_summary(parent_run_id: str, total_trials: int, 
                       best_result: Dict[str, Any], duration: float):
    """
    记录调优总结到父run
    
    Args:
        parent_run_id: 父级run ID
        total_trials: 总试验数
        best_result: 最佳结果
        duration: 总耗时（秒）
    """
    try:
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_metric('total_trials', total_trials)
            mlflow.log_metric('total_duration_sec', duration)
            mlflow.log_metric('avg_duration_per_trial', duration / total_trials if total_trials > 0 else 0)
            
            if best_result:
                mlflow.log_metric(f"best_{best_result.get('metric', 'metric')}", 
                                best_result.get('metric_value', 0))
                
                # 记录最佳参数（使用 log_param 而不是 log_text，避免 S3 上传）
                best_params = best_result.get('params', {})
                for key, value in best_params.items():
                    try:
                        # MLflow 参数长度限制为 500 字符
                        param_value = str(value)[:500]
                        mlflow.log_param(f'best_{key}', param_value)
                    except Exception:
                        pass  # 忽略记录失败
                
                # 记录最佳 run ID
                if 'run_id' in best_result:
                    mlflow.log_param('best_run_id', best_result['run_id'])
        
        print(f"✅ 调优总结已记录到 MLflow")
    except Exception as e:
        print(f"⚠️  记录调优总结失败: {e}")


def compare_runs(parent_run_id: str, metric: str = 'valid_loss',
                output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    对比所有runs的性能
    
    Args:
        parent_run_id: 父级run ID  
        metric: 对比指标
        output_dir: 输出目录（可选）
        
    Returns:
        对比结果DataFrame
    """
    client = mlflow.tracking.MlflowClient()
    
    runs = client.search_runs(
        experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )
    
    data = []
    for run in runs:
        row = {
            'run_id': run.info.run_id,
            'trial_idx': run.data.params.get('trial_idx', ''),
            metric: run.data.metrics.get(metric),
            **{k: v for k, v in run.data.params.items() if not k.startswith('base_')}
        }
        data.append(row)
    
    df = pd.DataFrame(data).sort_values(metric)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'runs_comparison.csv', index=False)
        print(f"📊 保存对比结果到: {output_dir / 'runs_comparison.csv'}")
    
    return df
