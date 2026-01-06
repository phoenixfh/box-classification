"""
FastAI超参数调优主脚本

支持网格搜索、随机搜索、贝叶斯优化（Optuna）
支持分布式多GPU训练
"""

import argparse
import yaml
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
from fastai.distributed import setup_distrib, teardown_distrib
import torch
import torch.distributed as dist

# 添加父目录到路径（确保可以导入 utils）
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import train_model
from utils.tuning import create_search_strategy, OptunaSearchStrategy
from utils.mlflow_tuning import (
    TuningExperimentManager, 
    select_best_run,
    analyze_parameter_importance,
    TuningCheckpoint,
    log_tuning_summary,
    compare_runs
)
from utils import is_main_process


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 验证必需字段
    if 'search_space' not in config:
        raise ValueError("配置文件必须包含 'search_space' 字段")
    
    if 'base_args' not in config:
        raise ValueError("配置文件必须包含 'base_args' 字段")
    
    return config


def merge_params(base_args: Dict[str, Any], trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """合并基础参数和trial参数"""
    merged = base_args.copy()
    
    # 过滤掉内部字段
    for key, value in trial_params.items():
        if not key.startswith('_'):
            merged[key] = value
    
    # 移除不应传递给 train_model 的参数
    exclude_keys = {'_tuning_metric', 'mlflow_uri'}
    for key in exclude_keys:
        merged.pop(key, None)
    
    return merged


def run_single_trial(trial_idx: int, trial_params: Dict[str, Any], 
                     base_args: Dict[str, Any], exp_manager: TuningExperimentManager
                     ) -> Optional[float]:
    """
    运行单个trial
    
    Args:
        trial_idx: trial索引
        trial_params: trial参数
        base_args: 基础参数
        exp_manager: 实验管理器
    
    Returns:
        验证指标值，如果失败返回None
    """
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"🔬 Trial {trial_idx + 1}")
        print(f"{'='*80}")
        
        # 显示参数
        print("参数:")
        for key, value in trial_params.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
    
    # 启动MLflow trial run（仅主进程）
    trial_run = None
    if is_main_process():
        trial_run = exp_manager.start_trial_run(trial_idx, trial_params)
    
    try:
        # 合并参数
        train_args = merge_params(base_args, trial_params)
        
        # 为每个 trial 设置唯一的模型保存路径
        project_name = train_args.get('project_name', 'hyperparameter-tuning')
        task_name = train_args.get('task_name', 'tuning')
        trial_task_name = f"{task_name}_trial_{trial_idx:03d}"
        train_args['task_name'] = trial_task_name  # 每个 trial 有独立的目录
        
        # 确保only_val=False（我们要训练）
        train_args['only_val'] = False
        
        # 默认不导出ONNX（调优时避免额外开销）
        if 'export_onnx' not in train_args:
            train_args['export_onnx'] = False
        
        # 调优模式：禁用模型上传到 MLflow（避免 S3 凭证问题）
        train_args['skip_mlflow_model_upload'] = True
        
        # 禁用 auto_resume（每个 trial 都是全新训练）
        train_args['auto_resume'] = False
        
        # 将 trial run ID 传递给训练（仅主进程）
        if is_main_process() and trial_run is not None:
            train_args['mlflow_parent_run_id'] = trial_run.info.run_id
        
        # 临时恢复 MLflow URI 用于训练
        import os
        mlflow_uri = base_args.get('mlflow_uri', 'http://192.168.16.130:5000/')
        mlflow_backup = os.environ.get('MLFLOW_TRACKING_URI', '')
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        
        try:
            # 分布式训练时，不在这里初始化（已在tune主函数初始化）
            # 只在单独调用 train_model 时才需要 setup_distrib
            train_model(**train_args)
        finally:
            # 恢复为空（禁用 S3）
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_backup
        
        # 获取指标值（从MLflow，仅主进程）
        if is_main_process():
            run = mlflow.get_run(trial_run.info.run_id)
            metric_name = base_args.get('_tuning_metric', 'valid_loss')
            
            # 调试：打印所有可用的指标
            available_metrics = list(run.data.metrics.keys())
            if is_main_process():
                print(f"📊 可用指标: {available_metrics}")
            
            metric_value = run.data.metrics.get(metric_name)
            
            if metric_value is None:
                print(f"⚠️ 未找到指标 '{metric_name}'，trial失败")
                print(f"   可用指标: {available_metrics}")
                return None
            
            print(f"\n✅ Trial完成: {metric_name} = {metric_value:.6f}")
            return metric_value
        else:
            # 非主进程返回None
            return None
        
    except Exception as e:
        if is_main_process():
            print(f"\n❌ Trial失败: {e}")
            import traceback
            traceback.print_exc()
        return None
    finally:
        if is_main_process() and trial_run is not None:
            exp_manager.end_trial_run()


def run_tuning(config: Dict[str, Any], resume_run_id: Optional[str] = None,
               dry_run: bool = False, distributed: bool = False):
    """
    运行调优主循环
    
    Args:
        config: 调优配置
        resume_run_id: 恢复的run ID（可选）
        dry_run: 仅显示参数组合，不实际训练
        distributed: 是否使用分布式训练
    """
    import os
    
    # 获取并设置 MLflow URI (仅主进程)
    if is_main_process():
        mlflow_uri = config.get('base_args', {}).get('mlflow_uri', 'http://192.168.16.130:5000/')
        mlflow.set_tracking_uri(mlflow_uri)
    
    # 禁用内置 MLflow 回调避免 S3 凭证问题
    mlflow_uri_backup = os.environ.get('MLFLOW_TRACKING_URI', '')
    if not dry_run:
        os.environ['MLFLOW_TRACKING_URI'] = ''
    
    try:
        _run_tuning_impl(config, resume_run_id, dry_run, 
                        config.get('base_args', {}).get('mlflow_uri', 'http://192.168.16.130:5000/'), 
                        distributed)
    finally:
        # 恢复 MLflow URI
        if mlflow_uri_backup:
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri_backup


def _run_tuning_impl(config: Dict[str, Any], resume_run_id: Optional[str] = None,
                     dry_run: bool = False, mlflow_uri: str = '', distributed: bool = False):
    """运行调优主循环的实际实现
    
    Args:
        config: 调优配置
        resume_run_id: 恢复的run ID
        dry_run: 仅显示参数组合
        mlflow_uri: MLflow tracking URI
        distributed: 是否使用分布式训练
    """
    
    # 分布式模式下初始化进程组（在任何分布式操作之前）
    if distributed and not dist.is_initialized():
        # 获取当前进程的 GPU
        import os
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        setup_distrib(gpu=local_rank)
        # 初始化后再检查是否为主进程
        is_main = is_main_process()
        if is_main:
            print(f"🔧 分布式环境已初始化 (GPUs: {torch.cuda.device_count()})")
    else:
        is_main = True  # 非分布式模式，当前进程就是主进程
    
    # 提取配置
    base_args = config['base_args']
    strategy_name = config.get('strategy', 'grid')
    metric = config.get('metric', 'valid_loss')
    mode = config.get('mode', 'minimize')
    
    # 将metric信息传递给train
    base_args['_tuning_metric'] = metric
    
    # 将distributed参数传递给train_model
    base_args['distributed'] = distributed
    
    # 仅主进程创建搜索策略和打印信息
    if is_main_process():
        print(f"\n🔍 搜索策略: {strategy_name}")
        if distributed:
            print(f"🚀 分布式训练: {torch.cuda.device_count()} GPUs")
    
    strategy = create_search_strategy(config)
    total_trials = strategy.get_total_trials()
    
    if is_main_process():
        print(f"📊 总试验数: {total_trials}")
    
    # Dry run模式 (仅主进程)
    if dry_run:
        if is_main_process():
            print(f"\n{'='*80}")
            print("🔍 Dry Run: 参数组合预览")
            print(f"{'='*80}\n")
            
            for idx, trial_params in enumerate(strategy.generate_trials()):
                print(f"Trial {idx + 1}:")
                for key, value in trial_params.items():
                    if not key.startswith('_'):
                        print(f"  {key}: {value}")
                print()
                
                if idx >= 9:  # 最多显示10个
                    print(f"... 还有 {total_trials - 10} 个组合\n")
                    break
            
            print(f"总计: {total_trials} 个参数组合")
        return
    
    # 项目名称
    project_name = base_args.get('project_name', 'hyperparameter-tuning')
    task_name = base_args.get('task_name', f"{strategy_name}-tuning")
    
    # 创建实验管理器 (仅主进程)
    exp_manager = None
    parent_run_id = None
    if is_main_process():
        exp_manager = TuningExperimentManager(project_name, task_name, config)
        
        # Checkpoint管理
        checkpoint_path = Path('runs') / project_name / task_name / 'tuning_checkpoint.json'
        checkpoint = TuningCheckpoint(checkpoint_path)
        
        # 恢复或新建
        completed_trials = set()
        if resume_run_id or checkpoint.exists():
            if resume_run_id:
                parent_run_id = resume_run_id
                print(f"🔄 从指定run恢复: {parent_run_id}")
            else:
                state = checkpoint.load()
                parent_run_id = state.get('parent_run_id')
                completed_trials = set(state.get('completed_trials', []))
                print(f"🔄 从checkpoint恢复: {len(completed_trials)} 个已完成")
            
            # TODO: 重新关联parent run
            # 目前简化处理，重新创建
            parent_run = exp_manager.start_parent_run()
            parent_run_id = parent_run.info.run_id
        else:
            parent_run = exp_manager.start_parent_run()
            parent_run_id = parent_run.info.run_id
    else:
        # 非主进程
        completed_trials = set()
        checkpoint = None
    
    # 主循环
    start_time = time.time()
    trial_results = []
    
    # 分布式模式下需要所有进程执行相同的trial
    # 主进程生成trial参数，然后广播给其他进程
    if distributed:
        # 收集所有trial参数 (仅主进程)
        all_trials = []
        if is_main_process():
            for trial_idx, trial_params in enumerate(strategy.generate_trials()):
                if trial_idx not in completed_trials:
                    all_trials.append((trial_idx, trial_params))
            print(f"📋 准备执行 {len(all_trials)} 个试验")
        
        # 广播trial数量
        trial_count = len(all_trials) if is_main_process() else 0
        trial_count = torch.tensor(trial_count, dtype=torch.long)
        if torch.cuda.is_available():
            trial_count = trial_count.cuda()
        dist.broadcast(trial_count, src=0)
        trial_count = trial_count.item()
        
        # 非主进程初始化空列表
        if not is_main_process():
            all_trials = [(None, None)] * trial_count
        
        # 逐个广播trial参数
        for i in range(trial_count):
            if is_main_process():
                trial_idx, trial_params = all_trials[i]
                # 序列化参数
                import pickle
                params_bytes = pickle.dumps((trial_idx, trial_params))
                params_size = len(params_bytes)
            else:
                params_size = 0
            
            # 广播参数大小
            size_tensor = torch.tensor(params_size, dtype=torch.long)
            if torch.cuda.is_available():
                size_tensor = size_tensor.cuda()
            dist.broadcast(size_tensor, src=0)
            params_size = size_tensor.item()
            
            # 广播参数内容
            if is_main_process():
                params_tensor = torch.ByteTensor(list(params_bytes))
            else:
                params_tensor = torch.ByteTensor(params_size)
            
            if torch.cuda.is_available():
                params_tensor = params_tensor.cuda()
            dist.broadcast(params_tensor, src=0)
            
            # 反序列化
            if not is_main_process():
                import pickle
                params_bytes = bytes(params_tensor.cpu().numpy())
                trial_idx, trial_params = pickle.loads(params_bytes)
                all_trials[i] = (trial_idx, trial_params)
        
        # 执行所有trial
        try:
            for trial_idx, trial_params in all_trials:
                # 运行trial (所有进程参与)
                metric_value = run_single_trial( trial_idx, trial_params, base_args, exp_manager)
                
                # 记录结果 (仅主进程)
                if is_main_process() and metric_value is not None:
                    # 确保 trial_params 可以 JSON 序列化
                    serializable_params = {}
                    for k, v in trial_params.items():
                        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                            serializable_params[k] = v
                        else:
                            # 转换为字符串
                            serializable_params[k] = str(v)
                    
                    trial_results.append({
                        'trial_idx': trial_idx,
                        'params': serializable_params,
                        'metric_value': metric_value
                    })
                    
                    # Optuna反馈
                    if isinstance(strategy, OptunaSearchStrategy):
                        strategy.report_result(trial_params, metric_value)
                    
                    # 保存checkpoint
                    completed_trials.add(trial_idx)
                    checkpoint.save({
                        'parent_run_id': parent_run_id,
                        'completed_trials': list(completed_trials),
                        'trial_results': trial_results
                    })
        except KeyboardInterrupt:
            if is_main_process():
                print("\n⚠️ 用户中断调优")
                print(f"已完成 {len(completed_trials)} 个试验")
                if parent_run_id:
                    print(f"恢复命令: python fastai/tune.py --resume {parent_run_id}")
        except Exception as e:
            if is_main_process():
                print(f"\n❌ 调优失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        # 非分布式模式
        try:
            for trial_idx, trial_params in enumerate(strategy.generate_trials()):
                # 跳过已完成的 (仅主进程检查)
                if is_main_process() and trial_idx in completed_trials:
                    print(f"⏭️  跳过已完成的 Trial {trial_idx + 1}")
                    continue
                
                # 运行trial (所有进程参与)
                metric_value = run_single_trial(
                    trial_idx, trial_params, base_args, exp_manager
                )
                
                # 记录结果 (仅主进程)
                if is_main_process() and metric_value is not None:
                    # 确保 trial_params 可以 JSON 序列化
                    serializable_params = {}
                    for k, v in trial_params.items():
                        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                            serializable_params[k] = v
                        else:
                            # 转换为字符串
                            serializable_params[k] = str(v)
                    
                    trial_results.append({
                        'trial_idx': trial_idx,
                        'params': serializable_params,
                        'metric_value': metric_value
                    })
                    
                    # Optuna反馈
                    if isinstance(strategy, OptunaSearchStrategy):
                        strategy.report_result(trial_params, metric_value)
                    
                    # 保存checkpoint
                    completed_trials.add(trial_idx)
                    checkpoint.save({
                        'parent_run_id': parent_run_id,
                        'completed_trials': list(completed_trials),
                        'trial_results': trial_results
                    })
        except KeyboardInterrupt:
            if is_main_process():
                print("\n⚠️ 用户中断调优")
                print(f"已完成 {len(completed_trials)} 个试验")
                if parent_run_id:
                    print(f"恢复命令: python fastai/tune.py --resume {parent_run_id}")
        except Exception as e:
            if is_main_process():
                print(f"\n❌ 调优失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 调优完成 (仅主进程处理)
    if is_main_process():
        duration = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ 调优完成!")
        print(f"{'='*80}")
        print(f"总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        print(f"完成试验: {len(trial_results)}/{total_trials}")
        
        # 选择最佳模型
        best_result = select_best_run(parent_run_id, metric, mode)
        
        # 参数重要性分析
        if len(trial_results) >= 10:
            output_dir = Path('runs') / project_name / task_name
            analyze_parameter_importance(parent_run_id, metric, output_dir)
            compare_runs(parent_run_id, metric, output_dir)
        
        # 先结束 parent run，再记录总结（避免 run 冲突）
        if exp_manager is not None:
            exp_manager.end_parent_run()
        
        # 记录总结
        log_tuning_summary(parent_run_id, len(trial_results), best_result, duration)
        
        # 清理checkpoint
        checkpoint.clear()
    
    # 确保所有进程完成
    if distributed:
        dist.barrier()
        if is_main_process():
            print("🔧 清理分布式环境...")
        teardown_distrib()


def main():
    parser = argparse.ArgumentParser(description='FastAI超参数调优')
    
    # 配置文件
    parser.add_argument('--config', type=str, required=True,
                       help='YAML配置文件路径')
    
    # 恢复
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复之前的调优run（提供parent run ID）')
    
    # Dry run
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示参数组合，不实际训练')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true',
                       help='启用多GPU分布式训练')
    
    # 参数覆盖
    parser.add_argument('--override', type=str, nargs='+',
                       help='覆盖配置参数，格式: key=value')
    
    args = parser.parse_args()
    
    # 加载配置（仅主进程打印）
    if is_main_process():
        print(f"📁 加载配置: {args.config}")
    config = load_config(args.config)
    
    # 参数覆盖
    if args.override:
        if is_main_process():
            print("\n⚙️ 参数覆盖:")
        for override in args.override:
            if '=' not in override:
                if is_main_process():
                    print(f"⚠️ 忽略无效覆盖: {override}")
                continue
            
            key, value = override.split('=', 1)
            
            # 尝试解析值
            try:
                # 尝试eval（支持数字、列表等）
                value = eval(value)
            except:
                # 保持字符串
                pass
            
            # 覆盖到base_args
            config['base_args'][key] = value
            if is_main_process():
                print(f"  {key} = {value}")
    
    # 显示配置摘要（仅主进程）
    if is_main_process():
        print(f"\n📋 配置摘要:")
        print(f"  策略: {config.get('strategy', 'grid')}")
        print(f"  试验数: {config.get('n_trials', 'all')}")
        print(f"  优化指标: {config.get('metric', 'valid_loss')} ({config.get('mode', 'minimize')})")
        print(f"  搜索参数: {', '.join(config['search_space'].keys())}")
    if args.distributed and is_main_process():
        import torch
        print(f"  分布式训练: {torch.cuda.device_count()} GPUs")
    
    # 运行调优
    run_tuning(config, resume_run_id=args.resume, dry_run=args.dry_run, distributed=args.distributed)


if __name__ == '__main__':
    main()
