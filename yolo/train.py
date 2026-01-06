"""YOLOv11 检测模型训练 - MLflow 集成"""
import warnings
import os
import sys
from pathlib import Path

# 添加父目录到路径以导入utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# 禁用 albumentations 的版本检查（避免网络超时警告）
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# 过滤掉 threadpoolctl 的警告
warnings.filterwarnings('ignore', category=UserWarning, module='threadpoolctl')
warnings.filterwarnings('ignore', message='Error fetching version info')

import mlflow
from ultralytics import YOLO
import argparse
import yaml
import pandas as pd
import shutil
from utils import is_main_process

# 导入 ONNX 导出函数
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from export_onnx import export_yolo_to_onnx


# export MLFLOW_TRACKING_URI=http://192.168.16.130:5000
# export AWS_ACCESS_KEY_ID=mlflow
# export AWS_SECRET_ACCESS_KEY=mlflow@SN
# export AWS_ENDPOINT_URL=http://192.168.16.130:9000
# export MLFLOW_S3_IGNORE_TLS=true


def create_mlflow_callbacks(run_id, mlflow_uri):
    """创建 MLflow 回调函数用于实时上报训练指标"""
    
    def on_fit_epoch_end(trainer):
        """每个 epoch 结束时上报指标到 MLflow"""
        # 先打印调试信息，确认回调被调用
        rank = getattr(trainer.args, 'rank', -1)
        print(f"\n🔔 回调触发 - Epoch {trainer.epoch}, Rank {rank}, 进程 PID {os.getpid()}")
        
        # 在分布式训练中，只有 rank 0 上报
        if rank != 0:
            print(f"   ⏭️  Rank {rank} 跳过上报（仅 Rank 0 上报）")
            return
            
        try:
            # 重新设置 MLflow（因为在子进程中）
            import mlflow as mlf
            mlf.set_tracking_uri(mlflow_uri)
            
            # 获取当前 epoch
            epoch = trainer.epoch
            
            # 调试信息
            print(f"\n🔔 MLflow Callback 被调用 - Epoch {epoch} (Rank {trainer.args.rank})")
            
            # 构建指标字典
            metrics = {}
            
            # 训练损失 (从 trainer.label_loss_items 获取)
            if hasattr(trainer, 'label_loss_items') and hasattr(trainer, 'tloss'):
                loss_items = trainer.label_loss_items(trainer.tloss, prefix="train")
                for k, v in loss_items.items():
                    metrics[k] = float(v)
            
            # 验证指标 (从 trainer.metrics 获取)
            if hasattr(trainer, 'metrics') and trainer.metrics:
                metric_dict = trainer.metrics.results_dict
                for k, v in metric_dict.items():
                    # 重命名指标以匹配 MLflow 习惯
                    if 'metrics/' in k:
                        metrics[k] = float(v)
                    elif k.startswith('val/'):
                        metrics[k] = float(v)
            
            # 学习率
            if hasattr(trainer, 'optimizer'):
                for i, param_group in enumerate(trainer.optimizer.param_groups):
                    metrics[f'lr/pg{i}'] = float(param_group['lr'])
            
            # 批量上报指标
            if metrics:
                print(f"📊 上报 {len(metrics)} 个指标到 MLflow (run_id={run_id}, step={epoch})")
                # 使用 run_id 上报到正确的 run
                with mlf.start_run(run_id=run_id):
                    mlf.log_metrics(metrics, step=epoch)
                print(f"✅ 指标上报成功")
            else:
                print(f"⚠️  没有找到可上报的指标")
                
        except Exception as e:
            print(f"❌ Callback 记录指标失败 (epoch {epoch}): {e}")
            import traceback
            traceback.print_exc()
    
    return {
        'on_fit_epoch_end': on_fit_epoch_end
    }

def train(
    data_yaml: str,
    model: str = 'yolo11s.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project_name: str = 'yolo-detection',
    task_name: str = 'experiment',
    mlflow_uri: str = 'http://192.168.16.130:5000/',
    overwrite: bool = False,
    mlflow_parent_run_id: str = None,  # NEW: 用于嵌套runs (调优模式)
    skip_mlflow_model_upload: bool = False,  # NEW: 跳过模型上传 (调优模式)
    model_size: str = 'medium',  # NEW: 模型规模 (nano/small/medium/large/xlarge)
    use_advanced_aug: bool = False,  # NEW: 使用高级数据增强
    use_hard_mining: bool = False,  # NEW: 使用困难样本挖掘
    **kwargs
):
    """
    训练 YOLOv11 检测模型并记录到 MLflow (宫颈细胞识别优化版本)
    
    Args:
        data_yaml: 数据集配置文件路径
        model: 模型名称或路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 单个GPU的批次大小（会自动乘以GPU数量作为总batch）
        project_name: MLflow 项目名称
        task_name: MLflow 运行名称
        mlflow_uri: MLflow 服务器地址
        overwrite: 强制重新开始训练（清空现有目录）
        model_size: 模型规模选择 (nano/small/medium/large/xlarge)
        use_advanced_aug: 启用高级数据增强（针对细胞识别优化）
        use_hard_mining: 启用困难样本挖掘
        **kwargs: 其他 YOLO 训练参数
    """
    
    # ============================================================
    # 宫颈细胞识别优化配置
    # ============================================================
    
    # 1. 模型规模映射
    model_size_map = {
        'nano': 'yolo11n.pt',
        'small': 'yolo11s.pt',
        'medium': 'yolo11m.pt',
        'large': 'yolo11l.pt',
        'xlarge': 'yolo11x.pt'
    }
    
    # 如果指定了 model_size，覆盖 model 参数
    if model_size and model_size.lower() in model_size_map:
        model = model_size_map[model_size.lower()]
        if is_main_process():
            print(f"\n🔧 模型规模: {model_size.upper()} -> {model}")
    
    # 2. 高级数据增强配置（针对细胞识别优化）
    if use_advanced_aug:
        if is_main_process():
            print(f"\n🎨 启用高级数据增强（宫颈细胞识别优化）")
        
        # 细胞图像特定的增强参数
        advanced_aug_params = {
            # 色彩增强 - 细胞染色变化
            'hsv_h': kwargs.get('hsv_h', 0.015),  # 色调变化（考虑染色差异）
            'hsv_s': kwargs.get('hsv_s', 0.7),    # 饱和度变化
            'hsv_v': kwargs.get('hsv_v', 0.4),    # 明度变化
            
            # 几何变换 - 细胞姿态多样性
            'degrees': kwargs.get('degrees', 0.0),      # 旋转（细胞无方向性）
            'translate': kwargs.get('translate', 0.1),  # 平移
            'scale': kwargs.get('scale', 0.5),          # 缩放（细胞大小变化）
            'shear': kwargs.get('shear', 0.0),          # 错切
            'perspective': kwargs.get('perspective', 0.0), # 透视变换
            'flipud': kwargs.get('flipud', 0.5),        # 垂直翻转
            'fliplr': kwargs.get('fliplr', 0.5),        # 水平翻转
            
            # Mosaic 和 MixUp
            'mosaic': kwargs.get('mosaic', 1.0),        # Mosaic 增强
            'mixup': kwargs.get('mixup', 0.1),          # MixUp 增强
            'copy_paste': kwargs.get('copy_paste', 0.1), # 复制粘贴增强
        }
        
        # 更新 kwargs
        kwargs.update(advanced_aug_params)
        
        if is_main_process():
            print(f"   色彩增强: hsv_h={advanced_aug_params['hsv_h']}, hsv_s={advanced_aug_params['hsv_s']}, hsv_v={advanced_aug_params['hsv_v']}")
            print(f"   几何变换: scale={advanced_aug_params['scale']}, translate={advanced_aug_params['translate']}")
            print(f"   混合增强: mosaic={advanced_aug_params['mosaic']}, mixup={advanced_aug_params['mixup']}, copy_paste={advanced_aug_params['copy_paste']}")
    
    # 3. 困难样本挖掘配置
    if use_hard_mining:
        if is_main_process():
            print(f"\n⛏️  启用困难样本挖掘")
        
        # 困难样本挖掘参数
        hard_mining_params = {
            # 增加小目标权重（细胞可能较小）
            'box': kwargs.get('box', 7.5),       # 边框损失权重
            'cls': kwargs.get('cls', 0.5),       # 分类损失权重
            'dfl': kwargs.get('dfl', 1.5),       # DFL损失权重
            
            # 优化器配置
            'optimizer': kwargs.get('optimizer', 'AdamW'),  # AdamW 对细节更敏感
            'lr0': kwargs.get('lr0', 0.001),                # 初始学习率
            'lrf': kwargs.get('lrf', 0.01),                 # 最终学习率系数
            'momentum': kwargs.get('momentum', 0.937),      # 动量
            'weight_decay': kwargs.get('weight_decay', 0.0005),  # 权重衰减
            
            # 学习率调度
            'cos_lr': kwargs.get('cos_lr', True),           # 余弦学习率
            'warmup_epochs': kwargs.get('warmup_epochs', 3.0),  # 预热轮数
            'warmup_momentum': kwargs.get('warmup_momentum', 0.8),
            'warmup_bias_lr': kwargs.get('warmup_bias_lr', 0.1),
            
            # 提高检测置信度阈值
            'conf': kwargs.get('conf', 0.25),    # 置信度阈值
            'iou': kwargs.get('iou', 0.7),       # NMS IoU阈值
            
            # 早停策略
            'patience': kwargs.get('patience', 100),  # 增加耐心值
        }
        
        # 更新 kwargs
        kwargs.update(hard_mining_params)
        
        if is_main_process():
            print(f"   损失权重: box={hard_mining_params['box']}, cls={hard_mining_params['cls']}, dfl={hard_mining_params['dfl']}")
            print(f"   优化器: {hard_mining_params['optimizer']}, lr0={hard_mining_params['lr0']}, lrf={hard_mining_params['lrf']}")
            print(f"   学习率策略: cos_lr={hard_mining_params['cos_lr']}, warmup_epochs={hard_mining_params['warmup_epochs']}")
            print(f"   检测阈值: conf={hard_mining_params['conf']}, iou={hard_mining_params['iou']}")
    
    # 4. 通用细胞识别优化
    cell_optimizations = {
        # 关闭 Mosaic 的时机（最后10个epoch）
        'close_mosaic': kwargs.get('close_mosaic', 10),
        
        # 混合精度训练
        'amp': kwargs.get('amp', True),
        
        # 多尺度训练
        'multi_scale': kwargs.get('multi_scale', True),
        
        # 保存最佳模型
        'save': kwargs.get('save', True),
        'save_period': kwargs.get('save_period', -1),  # -1表示只保存best和last
    }
    
    kwargs.update(cell_optimizations)
    
    if is_main_process():
        print(f"\n🔬 细胞识别通用优化:")
        print(f"   close_mosaic={cell_optimizations['close_mosaic']}, amp={cell_optimizations['amp']}")
        print(f"   multi_scale={cell_optimizations['multi_scale']}")
    
    # ============================================================
    
    # 计算GPU数量并调整batch size
    # 用户传入的 batch 作为单GPU的batch，需要乘以GPU数量
    device_str = kwargs.get('device', '0')
    if isinstance(device_str, str) and ',' in device_str:
        # 多GPU: "0,1,2,3" -> 4个GPU
        world_size = len(device_str.split(','))
    elif isinstance(device_str, (list, tuple)):
        world_size = len(device_str)
    else:
        # 单GPU
        world_size = 1
    
    # 保存用户设置的单GPU batch（用于显示）
    batch_per_gpu = batch
    # 计算总batch（YOLO会自动除以world_size，所以我们预先乘上去）
    total_batch = batch_per_gpu * world_size
    
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"🔧 Batch Size 自动调整")
        print(f"{'='*80}")
        print(f"检测到GPU数量: {world_size}")
        print(f"用户设置 (单GPU batch): {batch_per_gpu}")
        print(f"总batch (传递给YOLO): {total_batch}")
        print(f"YOLO内部计算 (每GPU实际batch): {total_batch} ÷ {world_size} = {batch_per_gpu}")
        print(f"{'='*80}\n")
    
    # 使用调整后的总batch
    batch = total_batch
    
    # 设置 MinIO/S3 访问凭据（用于 MLflow artifacts 存储）
    os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'mlflow@SN'
    os.environ['AWS_ENDPOINT_URL'] = 'http://192.168.16.130:9000'
    os.environ['AWS_REGION'] = ''
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
    
    # 设置 MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    
    # 处理已删除的实验
    try:
        mlflow.set_experiment(project_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted experiment" in str(e):
            if is_main_process():
                print(f"⚠️  实验 '{project_name}' 已被删除，正在恢复...")
            # 获取 MLflow 客户端
            client = mlflow.tracking.MlflowClient()
            # 查找已删除的实验
            exp = client.get_experiment_by_name(project_name)
            if exp and exp.lifecycle_stage == 'deleted':
                # 恢复实验
                client.restore_experiment(exp.experiment_id)
                if is_main_process():
                    print(f"✅ 已恢复实验 '{project_name}' (ID: {exp.experiment_id})")
                mlflow.set_experiment(project_name)
            else:
                if is_main_process():
                    print(f"❌ 无法恢复实验 '{project_name}'")
                raise
        else:
            raise
    
    # 确定性目录管理
    run_dir = Path('runs') / project_name / task_name
    last_pt = run_dir / 'weights' / 'last.pt'
    resume_training = False
    training_mode = "新训练"
    saved_args = None
    saved_epoch = -1
    
    # 检查目录和恢复逻辑
    if run_dir.exists():
        if overwrite:
            if is_main_process():
                print(f"🔄 强制覆盖现有训练，从头开始...")
            shutil.rmtree(run_dir)
            training_mode = "覆盖训练"
        elif last_pt.exists():
            # 发现 checkpoint，自动恢复训练
            resume_training = True
            training_mode = "恢复训练"
            
            try:
                import torch
                checkpoint = torch.load(last_pt, map_location='cpu', weights_only=False)
                saved_epoch = checkpoint.get('epoch', -1)
                
                # 检查是否需要增加 epochs
                if saved_epoch >= epochs - 1:
                    if is_main_process():
                        print(f"⚠️  检测到训练已完成 {saved_epoch + 1} 个 epoch")
                        print(f"⚠️  当前 --epochs={epochs}，训练无法继续")
                        print(f"💡 解决方法：")
                        print(f"   1. 增加 epochs 参数：--epochs {saved_epoch + 100}")
                        print(f"   2. 使用 --overwrite 从头开始训练")
                        print(f"   3. 删除 checkpoint 后重新训练")
                    raise ValueError(f"训练已完成，需要增加 epochs 或使用 --overwrite")
                
                if is_main_process():
                    print(f"🔄 检测到现有训练，从 epoch {saved_epoch + 1} 恢复到 epoch {epochs}...")
                
                # 读取训练参数
                if 'train_args' in checkpoint:
                    saved_args = checkpoint['train_args']
                    if is_main_process():
                        print(f"📋 从 checkpoint 加载训练参数...")
                else:
                    # 备用方案：从 args.yaml 读取
                    args_yaml_path = run_dir / 'args.yaml'
                    if args_yaml_path.exists():
                        with open(args_yaml_path, 'r') as f:
                            saved_args = yaml.safe_load(f)
                        if is_main_process():
                            print(f"📋 从 args.yaml 加载训练参数...")
                
                # 显示优化器状态
                if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                    opt_state = checkpoint['optimizer']
                    if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
                        current_lr = opt_state['param_groups'][0].get('lr', 'N/A')
                        if is_main_process():
                            print(f"📊 优化器当前学习率: {current_lr}")
            except Exception as e:
                if is_main_process():
                    print(f"🔄 检测到现有训练，继续训练...")
                    print(f"⚠️  读取训练参数失败: {e}")
                # 如果是我们主动抛出的错误，向上传播
                if isinstance(e, ValueError) and "训练已完成" in str(e):
                    raise
        else:
            if is_main_process():
                print(f"⚠️  目录存在但无检查点，将重新训练")
            shutil.rmtree(run_dir)
            training_mode = "重新训练"
    
    # 参数优先级：checkpoint > 命令行
    # 恢复训练时，优先使用 checkpoint 中的参数
    if resume_training and saved_args:
        if is_main_process():
            print(f"\n📋 恢复训练 - 从 checkpoint 加载参数:")
            print(f"{'参数':<20} {'Checkpoint值':<20} {'命令行值':<20} {'实际使用':<20}")
            print(f"{'-'*80}")
        
        # 从 checkpoint 恢复所有训练参数
        param_keys = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                      'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl', 
                      'optimizer', 'close_mosaic', 'amp', 'patience', 'cos_lr',
                      'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
                      'scale', 'mosaic', 'mixup', 'fliplr']
        
        for key in param_keys:
            checkpoint_val = saved_args.get(key)
            cmdline_val = kwargs.get(key)
            
            if checkpoint_val is not None:
                # 优先使用 checkpoint 中的值
                kwargs[key] = checkpoint_val
                if is_main_process():
                    if cmdline_val is not None and cmdline_val != checkpoint_val:
                        status = "⚠️  已忽略命令行"
                    else:
                        status = "✓ 已恢复"
                    print(f"{key:<20} {str(checkpoint_val):<20} {str(cmdline_val):<20} {str(checkpoint_val):<20} [{status}]")
        
        if is_main_process():
            print(f"{'-'*80}")
        
        # 基础参数对比（epochs, imgsz, batch 可能需要调整）
        for key in ['epochs', 'imgsz', 'batch']:
            checkpoint_val = saved_args.get(key)
            current_val = locals()[key]
            if checkpoint_val is not None:
                if checkpoint_val != current_val:
                    if is_main_process():
                        print(f"⚠️  {key}: checkpoint={checkpoint_val}, 命令行={current_val}, 使用命令行值（可能需要调整）")
        
        if is_main_process():
            print()
    
    # 确定最终使用的参数（用于显示）
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"🚀 YOLOv11 检测模型训练")
        print(f"{'='*80}")
        print(f"模型: {model}")
        print(f"数据集: {data_yaml}")
        print(f"训练模式: {training_mode}")
        print(f"MLflow 项目: {project_name}")
        print(f"实验名称: {task_name}")
        print(f"运行目录: {run_dir}")
        
        # 显示参数来源
        if resume_training and saved_args:
            print(f"参数来源: ✓ Checkpoint (从 epoch {saved_epoch + 1} 恢复)")
        else:
            print(f"参数来源: 命令行参数")
        
        print(f"\n📊 基础训练参数:")
        print(f"   epochs: {epochs}")
        print(f"   imgsz: {imgsz}")
        print(f"   batch (总batch): {batch}")
        print(f"   batch (每GPU): {batch_per_gpu}")
        print(f"   GPU数量: {world_size}")
        
        # 显示训练超参数（优先从 kwargs 获取，如果是 resume 模式，kwargs 已经包含 checkpoint 的值）
        print(f"\n📚 学习率与优化器:")
        lr_params = ['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                     'warmup_momentum', 'warmup_bias_lr', 'optimizer', 'cos_lr']
        for key in lr_params:
            value = kwargs.get(key, '默认')
            if value != '默认' or (resume_training and saved_args and saved_args.get(key) is not None):
                print(f"   {key}: {value}")
        
        # 显示损失权重
        print(f"\n⚖️  损失权重:")
        loss_params = ['box', 'cls', 'dfl']
        for key in loss_params:
            value = kwargs.get(key, '默认')
            if value != '默认':
                print(f"   {key}: {value}")
        
        # 显示数据增强参数
        print(f"\n🎨 数据增强:")
        aug_param_keys = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
                          'scale', 'mosaic', 'mixup', 'close_mosaic', 'fliplr']
        aug_params = {k: kwargs.get(k) for k in aug_param_keys if kwargs.get(k) is not None}
        if aug_params:
            for key, value in aug_params.items():
                print(f"   {key}: {value}")
        else:
            print(f"   使用默认值")
        
        # 显示其他参数
        known_keys = set(['lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 
                          'warmup_momentum', 'warmup_bias_lr', 'optimizer', 'cos_lr',
                          'box', 'cls', 'dfl', 'patience', 'amp', 'project', 'name'] + aug_param_keys)
        other_params = {k: v for k, v in kwargs.items() if k not in known_keys and v is not None}
        if other_params:
            print(f"\n⚙️  其他参数:")
            for key, value in other_params.items():
                print(f"   {key}: {value}")
        
        print(f"{'='*80}\n")
    
    # 开始MLflow追踪
    if is_main_process():
        print(f"\n📊 启动 MLflow 追踪...")
        print(f"   模式: {training_mode}")
        print(f"   项目: {project_name}")
        print(f"   任务: {task_name}")
    
    # 处理嵌套 run (调优模式)
    if mlflow_parent_run_id:
        # 调优模式：重用父 trial run
        existing_run = mlflow.active_run()
        if existing_run and existing_run.info.run_id == mlflow_parent_run_id:
            run = existing_run
            if is_main_process():
                print(f"   ✅ 使用调优 Trial Run: {mlflow_parent_run_id}")
        else:
            # 非主进程或 run 不活跃：仅主进程创建 run
            if is_main_process():
                print(f"   ⚠️  警告: 父run不是活跃状态，创建新run")
                run = mlflow.start_run(run_name=task_name)
            else:
                # 非主进程：创建一个虚拟 context，不实际创建 MLflow run
                from contextlib import nullcontext
                run = nullcontext()
    else:
        # 正常训练模式：仅主进程创建 run
        if is_main_process():
            run = mlflow.start_run(run_name=task_name)
            print(f"   ✅ 创建新的 MLflow Run")
        else:
            # 非主进程：不创建 MLflow run
            from contextlib import nullcontext
            run = nullcontext()
    
    with run:
        # 获取 run_id（仅主进程或有效 run）
        run_id = None
        if hasattr(run, 'info'):
            run_id = run.info.run_id
            if is_main_process():
                print(f"📊 MLflow Run ID: {run_id}")
        
        # 记录训练参数（仅主进程）
        if is_main_process() and run_id:
            params = {
                'model': model,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch_total': batch,
                'batch_per_gpu': batch_per_gpu,
                'world_size': world_size,
                'data_yaml': data_yaml,
                'model_size': model_size,
                'use_advanced_aug': use_advanced_aug,
                'use_hard_mining': use_hard_mining,
            }
            
            # 添加其他训练参数
            for key, value in kwargs.items():
                if value is not None and key not in ['project', 'name']:
                    params[f'train/{key}'] = value
            
            mlflow.log_params(params)
        
        # 读取数据集信息
        try:
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # 记录数据集信息（仅主进程）
            if is_main_process() and run_id:
                mlflow.log_param('dataset/classes', str(data_config.get('names', [])))
                mlflow.log_param('dataset/path', data_config.get('path', ''))
            
            if is_main_process():
                print(f"📁 数据集信息:")
                print(f"   类别数: {len(data_config.get('names', []))}")
                print(f"   类别: {data_config.get('names', [])}")
                print(f"   数据路径: {data_config.get('path', '')}")
                print()
        except Exception as e:
            if is_main_process():
                print(f"⚠️  读取数据集配置失败: {e}")
                print(f"当前路径: {os.getcwd()}")
            exit(-1)
        
        # 加载模型
        if is_main_process():
            print(f"🔧 加载模型: {model}")
        yolo_model = YOLO(model)
        
        # 训练（YOLO 会自动记录到 runs/ 目录）
        if is_main_process():
            print(f"\n{'='*80}")
            print(f"🏃 开始训练...")
            print(f"{'='*80}\n")
        
        # 禁用 Ultralytics 内置的 MLflow 集成
        from ultralytics.utils import SETTINGS
        mlflow_enabled_backup = SETTINGS.get('mlflow', True)
        SETTINGS['mlflow'] = False
        
        mlflow_uri_backup = os.environ.get('MLFLOW_TRACKING_URI', '')
        os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.16.130:5000'
        os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'mlflow@SN'
        os.environ['AWS_ENDPOINT_URL'] = 'http://192.168.16.130:9000'
        os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

        # 构建训练参数
        train_kwargs = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'project': str(run_dir.parent),
            'name': run_dir.name,
            'exist_ok': True,
            **kwargs
        }
        
        # 如果需要恢复训练，设置 resume
        if resume_training:
            train_kwargs['resume'] = True
            # 重新加载模型用于恢复
            yolo_model = YOLO(str(last_pt))
        
        # 添加 MLflow 回调（实时上报指标）
        # 必须在模型最终加载后注册，避免被覆盖
        if run_id:
            print(f"📊 注册 MLflow 回调函数（实时上报训练指标）")
            print(f"   Run ID: {run_id}")
            print(f"   MLflow URI: {mlflow_uri}")
            mlflow_callbacks = create_mlflow_callbacks(run_id, mlflow_uri)
            for event, func in mlflow_callbacks.items():
                yolo_model.add_callback(event, func)
            print(f"   ✅ 已注册 {len(mlflow_callbacks)} 个回调事件")
        
        results = yolo_model.train(**train_kwargs)
        
        # 恢复 MLflow 设置
        SETTINGS['mlflow'] = mlflow_enabled_backup
        if mlflow_uri_backup:
            os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri_backup
        
        # 获取训练结果路径
        # YOLO 某些版本 train() 可能返回 None，使用我们预设的 run_dir
        save_dir = Path(results.save_dir) if results and hasattr(results, 'save_dir') else run_dir
        if is_main_process():
            print(f"\n✅ 训练完成！结果保存在: {save_dir}")
        
        # 记录最终指标（仅主进程）
        if is_main_process():
            print(f"\n📊 记录训练指标到 MLflow...")
            
        if (save_dir / 'results.csv').exists() and is_main_process() and run_id:
            results_df = pd.read_csv(save_dir / 'results.csv')
            results_df = results_df.fillna(0)  # 填充 NaN
            
            # 记录最后一轮的指标
            last_metrics = results_df.iloc[-1]
            final_metrics = {
                'final/mAP50': float(last_metrics.get('metrics/mAP50(B)', 0)),
                'final/mAP50-95': float(last_metrics.get('metrics/mAP50-95(B)', 0)),
                'final/precision': float(last_metrics.get('metrics/precision(B)', 0)),
                'final/recall': float(last_metrics.get('metrics/recall(B)', 0)),
                'final/box_loss': float(last_metrics.get('train/box_loss', 0)),
                'final/cls_loss': float(last_metrics.get('train/cls_loss', 0)),
                'final/dfl_loss': float(last_metrics.get('train/dfl_loss', 0)),
            }
            mlflow.log_metrics(final_metrics)
            
            print(f"   ✅ 最终指标:")
            print(f"      mAP@0.5: {final_metrics['final/mAP50']:.4f}")
            print(f"      mAP@0.5:0.95: {final_metrics['final/mAP50-95']:.4f}")
            print(f"      Precision: {final_metrics['final/precision']:.4f}")
            print(f"      Recall: {final_metrics['final/recall']:.4f}")
            
            # 上传完整结果（跳过调优模式）
            if not skip_mlflow_model_upload:
                mlflow.log_artifact(str(save_dir / 'results.csv'), 'training_results')
        elif is_main_process() and (save_dir / 'results.csv').exists() == False:
                print(f"   ⚠️  未找到 results.csv")
        
        # 上传模型和 artifacts（仅主进程且非调优模式）
        if is_main_process() and run_id and not skip_mlflow_model_upload:
            print(f"\n📦 上传模型到 MLflow...")
            best_model = save_dir / 'weights' / 'best.pt'
            last_model = save_dir / 'weights' / 'last.pt'
            
            if best_model.exists():
                mlflow.log_artifact(str(best_model), 'models')
                print(f"   ✅ best.pt 已上传")
                
                # 导出 ONNX 格式
                print(f"\n🔄 导出 best.pt 为 ONNX 格式...")
                try:
                    onnx_path = best_model.with_suffix('.onnx')
                    # 使用专用的导出函数（动态 batch，固定其他维度）
                    export_yolo_to_onnx(
                        model_path=str(best_model),
                        imgsz=imgsz,
                        output_path=str(onnx_path)
                    )
                    
                    if onnx_path.exists():
                        print(f"   ✅ ONNX 导出成功: {onnx_path.name}")
                        # 上传 ONNX 模型到 MLflow
                        mlflow.log_artifact(str(onnx_path), 'models')
                        print(f"   ✅ ONNX 模型已上传到 MLflow")
                    else:
                        print(f"   ⚠️  ONNX 文件未找到")
                except Exception as e:
                    print(f"   ⚠️  ONNX 导出失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            if last_model.exists():
                mlflow.log_artifact(str(last_model), 'models')
                print(f"   ✅ last.pt 已上传")
            
            # 上传可视化图表
            print(f"\n📈 上传可视化图表...")
            plots = ['confusion_matrix.png', 'results.png', 'PR_curve.png', 
                    'F1_curve.png', 'labels.jpg', 'labels_correlogram.jpg']
            
            uploaded_plots = 0
            for plot in plots:
                plot_path = save_dir / plot
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path), 'plots')
                    uploaded_plots += 1
            
            print(f"   ✅ 上传了 {uploaded_plots} 个图表")
            # 上传训练配置
            args_yaml = save_dir / 'args.yaml'
            if args_yaml.exists():
                mlflow.log_artifact(str(args_yaml), 'config')
        
        if is_main_process():
            print(f"\n{'='*80}")
            print(f"✅ 训练完成并记录到 MLflow！")
            print(f"{'='*80}")
            if run_id:
                print(f"📊 MLflow Run ID: {run_id}")
                print(f"📁 模型保存位置: {save_dir}")
                if hasattr(run, 'info'):
                    print(f"🌐 MLflow UI: {mlflow_uri}#/experiments/{run.info.experiment_id}/runs/{run_id}")
            else:
                print(f"📁 模型保存位置: {save_dir}")
            print(f"{'='*80}\n")
        
        return results, run_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLOv11 检测模型训练 - MLflow 集成 (支持断点续训)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 宫颈细胞识别 - 高精度训练
  python yolo/train.py \\
      --data docs/cell-core/cell.yaml \\
      --model yolo11n.pt \\
      --epochs 5000 \\
      --imgsz 1024 \\
      --batch 8 \\
      --device 0,1,2,3,4 \\
      --model_size medium \\
      --use_advanced_aug \\
      --use_hard_mining \\
      --project_name cell-box-modify \\
      --task_name high-accuracy
  
  # 基础训练 (单GPU)
  python yolo/train.py --data pk-dataset.yaml --model yolo11s.pt --epochs 100 --batch 16

  # 多GPU训练 (batch参数为单GPU的batch，自动乘以GPU数)
  python yolo/train.py \\
      --data dna-classify-cluster.yaml \\
      --model yolo11m.pt \\
      --epochs 200 \\
      --batch 4 \\          # 单GPU batch=4, 8个GPU总batch=32
      --device 0,1,2,3,4,5,6,7 \\
      --project_name dna-detection \\
      --task_name yolo11m-exp1
  
  # 恢复训练（自动检测 last.pt）
  python yolo/train.py \\
      --data dataset.yaml \\
      --project_name my-project \\
      --task_name exp-001 \\
      --epochs 300
  
  # 强制重新训练（忽略已有检查点）
  python yolo/train.py \\
      --data dataset.yaml \\
      --project_name my-project \\
      --task_name exp-001 \\
      --epochs 100 \\
      --overwrite

宫颈细胞识别优化:
  --model_size: 选择模型规模 (nano/small/medium/large/xlarge)
  --use_advanced_aug: 启用针对细胞图像的高级数据增强
                      - 色彩增强：模拟染色差异
                      - 几何变换：细胞姿态多样性
                      - 混合增强：Mosaic, MixUp, Copy-Paste
  --use_hard_mining: 启用困难样本挖掘
                     - Focal Loss关注困难样本
                     - 增加小目标检测权重
                     - 优化学习率调度
                     - 提高检测置信度阈值

重要说明:
  --batch 参数代表"单个GPU"的batch size，程序会自动乘以GPU数量
  例如: --batch 4 --device 0,1,2,3,4,5,6,7
       实际总batch = 4 × 8 = 32
       每个GPU分配 = 32 ÷ 8 = 4 ✓
      
支持所有 YOLO 训练参数，如: device, workers, patience, optimizer, lr0, lrf, 
hsv_h, hsv_s, hsv_v, degrees, translate, scale, mosaic, mixup, close_mosaic 等

目录管理:
  - 训练结果保存在: runs/{project_name}/{task_name}/
  - 如果目录存在且包含 last.pt，自动恢复训练
  - 使用 --overwrite 强制从头开始
        """
    )
    
    # 必需参数
    parser.add_argument('--data', type=str, required=True, 
                       help='数据集 YAML 配置文件路径')
    
    # 基础参数
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                       help='YOLO 模型 (默认: yolo11s.pt, 会被 --model_size 覆盖)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--batch', type=int, default=16, 
                       help='单GPU的批次大小 (默认: 16, 会自动乘以GPU数量)')
    
    # 宫颈细胞识别专用参数
    parser.add_argument('--model_size', type=str, 
                       choices=['nano', 'small', 'medium', 'large', 'xlarge'],
                       help='模型规模 (nano/small/medium/large/xlarge)')
    parser.add_argument('--use_advanced_aug', action='store_true',
                       help='启用高级数据增强（针对细胞识别优化）')
    parser.add_argument('--use_hard_mining', action='store_true',
                       help='启用困难样本挖掘')
    
    # MLflow 参数
    parser.add_argument('--project_name', type=str, default='yolo-detection',
                       help='MLflow 项目名称 (默认: yolo-detection)')
    parser.add_argument('--task_name', type=str, default='experiment',
                       help='MLflow 运行名称 (默认: experiment)')
    parser.add_argument('--mlflow_uri', type=str, 
                       default='http://192.168.16.130:5000/',
                       help='MLflow 服务器地址')
    
    # 目录管理参数
    parser.add_argument('--overwrite', action='store_true',
                       help='强制重新开始训练，清空现有目录（默认: False，自动恢复训练）')
    
    # 解析已知参数，其余参数传递给 YOLO
    args, unknown = parser.parse_known_args()
    
    # 解析未知参数 (YOLO 参数)
    yolo_kwargs = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--'):
            key = arg[2:]
            # 检查下一个是否是值
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                # 尝试转换类型
                try:
                    # 尝试转为 int
                    yolo_kwargs[key] = int(value)
                except ValueError:
                    try:
                        # 尝试转为 float
                        yolo_kwargs[key] = float(value)
                    except ValueError:
                        # 布尔值
                        if value.lower() in ['true', 'false']:
                            yolo_kwargs[key] = value.lower() == 'true'
                        else:
                            yolo_kwargs[key] = value
                i += 2
            else:
                # 无值的参数，当作 True
                yolo_kwargs[key] = True
                i += 1
        else:
            i += 1
    
    # 打印所有参数
    if yolo_kwargs:
        print(f"\n📝 额外 YOLO 参数: {yolo_kwargs}\n")
    
    # 训练
    train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project_name=args.project_name,
        task_name=args.task_name,
        mlflow_uri=args.mlflow_uri,
        overwrite=args.overwrite,
        model_size=args.model_size,
        use_advanced_aug=args.use_advanced_aug,
        use_hard_mining=args.use_hard_mining,
        **yolo_kwargs
    )
