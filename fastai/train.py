from fastai.vision.all import *
from fastai.callback.all import *
from fastai.distributed import *
from pathlib import Path
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.optim import SGD as TorchSGD, Adam as TorchAdam, AdamW as TorchAdamW, RMSprop as TorchRMSprop
import argparse
import sys
import traceback
from fastai.callback.tracker import TrackerCallback
import mlflow
# import mlflow.pytorch
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
import warnings
import numpy as np
# from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from functools import partial
from accelerate.utils import write_basic_config, DistributedDataParallelKwargs

write_basic_config()

# 添加项目根目录到 sys.path（确保可以导入 utils）
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入自定义模型模块（在 fastai/models 目录下）
# 使用 importlib 来避免路径冲突
try:
    import importlib.util
    _models_path = Path(__file__).parent / 'models' / '__init__.py'
    _spec = importlib.util.spec_from_file_location('custom_models', _models_path)
    _custom_models = importlib.util.module_from_spec(_spec)
    sys.modules['custom_models'] = _custom_models  # 注册到 sys.modules
    _spec.loader.exec_module(_custom_models)
    
    get_model = _custom_models.get_model
    list_models = _custom_models.list_models
    is_custom_model = _custom_models.is_custom_model
    CUSTOM_MODELS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  自定义模型模块导入失败: {e}")
    print("   将仅使用fastai内置模型")
    CUSTOM_MODELS_AVAILABLE = False
    is_custom_model = lambda x: False

# 过滤 sklearn 的 UndefinedMetricWarning
warnings.filterwarnings('ignore', message='.*Precision is ill-defined.*')
warnings.filterwarnings('ignore', message='.*Recall is ill-defined.*')
warnings.filterwarnings('ignore', message='.*F1 score is ill-defined.*')

# 过滤 pandas FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Series.__getitem__.*')
warnings.filterwarnings('ignore', message='.*treating keys as positions.*')

# 过滤 DDP gradient strides 警告（这是 PyTorch 内部性能提示，不影响训练）
warnings.filterwarnings('ignore', message='.*Grad strides do not match bucket view strides.*')

# 注意: threadpoolctl 的 AttributeError 是 ctypes callback 中的异常，
# 无法通过 Python 的 warnings 模块抑制。这是 Python 3.12 的已知兼容性问题，
# 不影响训练功能。如需抑制，可以：
# 1. 升级 threadpoolctl: pip install -U threadpoolctl
# 2. 或在启动时重定向 stderr: python train.py 2>/dev/null

# 导入工具模块
from utils import (
    is_main_process, setup_mlflow,
    get_segmentation_dls,
    DiceMetric, CombinedLoss,
    YOLOv11LRScheduler, LoadOptimizerStateCallback, ResumeEpochCallback,
    EarlyStoppingWithEvalCallback, SaveModelWithEpochCallback,
    MLflowMetricsCallback,
    DistributedValidationDiagnosticCallback,
)

def print_model_structure(model, model_name='Model', max_depth=5, input_size=(1, 3, 224, 224), show_shape=True):
    """
    通用模型结构打印函数，支持所有类型的模型（Timm、YOLO、自定义等）
    
    参数:
        model: PyTorch模型
        model_name: 模型名称
        max_depth: 最大递归深度（避免过深）
        input_size: 输入张量大小 (batch, channels, height, width)
        show_shape: 是否显示每层的输入/输出shape
    
    特点:
        - 自动递归展开所有层
        - 树形结构显示层级关系
        - 自动提取层参数信息
        - 显示每层的输入/输出shape（可选）
        - 统一格式，支持所有模型类型
    """
    
    # 如果需要显示shape，先通过forward hook收集信息
    shape_info = {}
    hooks = []
    
    if show_shape:
        def hook_fn(name):
            def hook(module, input, output):
                try:
                    # 获取输入shape
                    if isinstance(input, tuple) and len(input) > 0:
                        in_shape = tuple(input[0].shape) if hasattr(input[0], 'shape') else None
                    else:
                        in_shape = tuple(input.shape) if hasattr(input, 'shape') else None
                    
                    # 获取输出shape
                    if isinstance(output, tuple):
                        out_shape = tuple(output[0].shape) if hasattr(output[0], 'shape') else None
                    else:
                        out_shape = tuple(output.shape) if hasattr(output, 'shape') else None
                    
                    shape_info[name] = {
                        'input': in_shape,
                        'output': out_shape
                    }
                except:
                    pass
            return hook
        
        # 注册hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只对叶子节点注册
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 执行一次forward获取shape信息
        try:
            import torch
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
        except Exception as e:
            print(f"⚠️  无法获取shape信息: {e}")
            show_shape = False
        finally:
            # 移除hooks
            for hook in hooks:
                hook.remove()
    
    # 打印表头
    print(f"\n{'='*150}")
    print(f"{model_name} - 模型结构详解")
    print(f"{'='*150}")
    if show_shape:
        print(f"{'idx':<5} {'layer_name':<45} {'type':<25} {'params':>12} {'input_shape':<25} {'output_shape':<25} {'details':<15}")
        print(f"{'-'*150}")
    else:
        print(f"{'idx':<5} {'layer_name':<50} {'type':<30} {'params':>12} {'details':<30}")
        print(f"{'-'*120}")
    
    layer_idx = 0
    total_params = 0
    total_trainable = 0
    layer_count = 0
    
    def format_number(num):
        """格式化数字显示"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num/1_000:.2f}K"
        else:
            return str(num)
    
    def extract_layer_info(module):
        """提取层的关键信息"""
        info = []
        
        # 卷积层
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            info.append(f"C:{module.in_channels}→{module.out_channels}")
            if hasattr(module, 'kernel_size'):
                k = module.kernel_size
                k_str = f"{k[0]}" if isinstance(k, (tuple, list)) else f"{k}"
                info.append(f"K:{k_str}")
            if hasattr(module, 'stride'):
                s = module.stride
                s_val = s[0] if isinstance(s, (tuple, list)) else s
                if s_val > 1:
                    info.append(f"S:{s_val}")
            if hasattr(module, 'padding'):
                p = module.padding
                p_val = p[0] if isinstance(p, (tuple, list)) else p
                if p_val > 0:
                    info.append(f"P:{p_val}")
        
        # 全连接层
        elif hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            info.append(f"FC:{module.in_features}→{module.out_features}")
        
        # 归一化层
        elif hasattr(module, 'num_features'):
            info.append(f"Norm:{module.num_features}")
        elif hasattr(module, 'normalized_shape'):
            shape = module.normalized_shape
            if isinstance(shape, (tuple, list)):
                shape_str = f"{shape[0]}" if len(shape) == 1 else f"{shape}"
            else:
                shape_str = f"{shape}"
            info.append(f"Norm:{shape_str}")
        
        # Pooling层
        elif hasattr(module, 'kernel_size') and 'Pool' in module.__class__.__name__:
            k = module.kernel_size
            k_str = f"{k[0]}" if isinstance(k, (tuple, list)) else f"{k}"
            info.append(f"Pool:K{k_str}")
            if hasattr(module, 'stride'):
                s = module.stride
                s_val = s[0] if isinstance(s, (tuple, list)) else s
                if s_val and s_val > 1:
                    info.append(f"S:{s_val}")
        
        # Dropout
        elif hasattr(module, 'p') and 'Dropout' in module.__class__.__name__:
            info.append(f"Drop:{module.p:.2f}")
        
        # Embedding
        elif hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
            info.append(f"Emb:{module.num_embeddings}×{module.embedding_dim}")
        
        # RNN/LSTM/GRU
        elif hasattr(module, 'input_size') and hasattr(module, 'hidden_size'):
            info.append(f"RNN:{module.input_size}→{module.hidden_size}")
            if hasattr(module, 'num_layers'):
                info.append(f"L:{module.num_layers}")
        
        # Attention (Transformer)
        elif hasattr(module, 'embed_dim'):
            info.append(f"Attn:D{module.embed_dim}")
            if hasattr(module, 'num_heads'):
                info.append(f"H:{module.num_heads}")
        
        return ", ".join(info) if info else ""
    
    def format_shape(shape):
        """格式化shape信息"""
        if shape is None:
            return "-"
        # 只显示 (C, H, W) 或关键维度，省略batch
        if len(shape) == 4:  # (B, C, H, W)
            return f"({shape[1]},{shape[2]},{shape[3]})"
        elif len(shape) == 3:  # (B, C, L) 或 (C, H, W)
            return f"({shape[1]},{shape[2]})"
        elif len(shape) == 2:  # (B, D)
            return f"({shape[1]})"
        else:
            return str(shape)
    
    def print_layers(module, prefix='', depth=0):
        """递归打印所有层"""
        nonlocal layer_idx, total_params, total_trainable, layer_count
        
        # 深度限制
        if depth >= max_depth:
            return
        
        # 遍历所有子模块
        for name, child in module.named_children():
            # 计算参数量
            params = sum(p.numel() for p in child.parameters())
            trainable_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            # 获取模块类型（简化名称）
            module_type = child.__class__.__name__
            
            # 提取详细信息
            details = extract_layer_info(child)
            
            # 构建层名称（带缩进）
            indent = "  " * depth
            if depth == 0:
                layer_name = f"{name}"
            else:
                layer_name = f"{indent}└─ {name}"
            
            # 判断是否显示该层
            # 1. 有参数的层必须显示
            # 2. 顶层容器必须显示
            # 3. 重要的无参数层也显示（Pool, Dropout, Activation等）
            is_leaf = len(list(child.children())) == 0
            is_important_container = depth <= 1 and not is_leaf
            is_important_layer = is_leaf and (
                'Pool' in module_type or 
                'Dropout' in module_type or 
                'Activation' in module_type or
                'ReLU' in module_type or
                'Sigmoid' in module_type or
                'Tanh' in module_type or
                'Softmax' in module_type or
                'Flatten' in module_type or
                'Identity' in module_type
            )
            
            show_layer = params > 0 or is_important_container or is_important_layer
            
            if show_layer:
                # 获取shape信息
                full_name = f"{prefix}.{name}" if prefix else name
                shapes = shape_info.get(full_name, {'input': None, 'output': None})
                
                # 如果层有参数但不可训练，标记出来
                frozen_mark = ""
                if params > 0 and trainable_params == 0:
                    frozen_mark = " [FROZEN]"
                
                if show_shape:
                    # 格式化输出（带shape）
                    layer_name_str = f"{layer_name:<45}"
                    type_str = f"{module_type:<25}"
                    params_str = f"{format_number(params):>12}"
                    in_shape_str = f"{format_shape(shapes['input']):<25}"
                    out_shape_str = f"{format_shape(shapes['output']):<25}"
                    details_str = f"{details:<15}"
                    
                    print(f"{layer_idx:<5} {layer_name_str} {type_str} {params_str} {in_shape_str} {out_shape_str} {details_str}{frozen_mark}")
                else:
                    # 格式化输出（不带shape）
                    layer_name_str = f"{layer_name:<50}"
                    type_str = f"{module_type:<30}"
                    params_str = f"{format_number(params):>12}"
                    details_str = f"{details:<30}"
                    
                    print(f"{layer_idx:<5} {layer_name_str} {type_str} {params_str} {details_str}{frozen_mark}")
                
                # 累计统计
                if params > 0:
                    total_params += params
                    total_trainable += trainable_params
                    layer_count += 1
                
                layer_idx += 1
            
            # 递归处理子模块
            if len(list(child.children())) > 0 and depth < max_depth - 1:
                print_layers(child, f"{prefix}.{name}" if prefix else name, depth + 1)
    
    # 开始打印
    try:
        print_layers(model)
    except Exception as e:
        print(f"⚠️  打印模型结构时出错: {e}")
        print(f"   尝试使用备用方式...")
        # 备用方式：使用named_modules
        for idx, (name, module) in enumerate(model.named_modules()):
            if len(list(module.children())) == 0:  # 只显示叶子节点
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    module_type = module.__class__.__name__
                    details = extract_layer_info(module)
                    print(f"{idx:<5} {name:<50} {module_type:<30} {format_number(params):>12} {details:<30}")
                    total_params += params
                    layer_count += 1
    
    # 打印统计信息
    sep_line = '='*150 if show_shape else '='*120
    print(sep_line)
    print(f"总计: {layer_count} 层")
    print(f"参数: {format_number(total_params)} ({total_params:,}) - 可训练: {format_number(total_trainable)} ({total_trainable:,})")
    
    # 估算模型大小
    model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
    print(f"模型大小: ~{model_size_mb:.2f} MB (float32)")
    
    if show_shape:
        print(f"输入尺寸: {input_size}")
    
    print(sep_line)
    print()
    # 估算模型大小和FLOPs
    model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
    estimated_flops = total_params * 2 / 1e9  # 粗略估算
    
    print(f"模型大小: ~{model_size_mb:.2f} MB (float32)")
    print(f"估算FLOPs: ~{estimated_flops:.2f} GFLOPs")
    print(f"{'='*120}\n")

# mlflow.config.enable_system_metrics_logging()
# mlflow.config.set_system_metrics_sampling_interval(1)
# mlflow.pytorch.autolog(log_models=False)  # 启用autolog但禁用自动模型保存，避免与手动log_model冲突

def call_evaluation_script(
    learn,
    model_path,
    mlflow_run_id,
    project_name='default'
):
    """
    调用独立的评估模块（复用训练数据，统一使用直接调用方式）
    
    Args:
        learn: FastAI Learner 对象（复用训练时的数据）
        model_path: 模型路径（用于加载best权重）
        mlflow_run_id: MLflow运行ID
        project_name: 项目名称（用于组织输出目录）
    """
    
    try:
        print("\n📊 调用评估模块（复用训练数据）...")
        
        # 动态导入evaluate模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate", 
            Path(__file__).parent / "evaluate.py"
        )
        evaluate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate_module)
        
        # 构建输出目录
        output_dir = f"./evaluation_results/{project_name}/{Path(model_path).parent.name}"
        
        # 直接调用评估函数（传入learn对象，复用数据）
        # evaluate.py内部会清理分布式环境
        evaluate_module.evaluate_with_learner(
            learn=learn,
            model_path=str(model_path),
            output_dir=output_dir,
            mlflow_run_id=mlflow_run_id
        )
        
        print("   ✅ 评估完成")
        
    except Exception as e:
        print(f"⚠️ 评估失败: {e}")
        import traceback
        traceback.print_exc()


def load_classification_data(data_path, train_size=None, val_size=None, mlflow_run=None):
    """
    加载分类数据集，返回 train_df 和 valid_df
    
    Args:
        data_path: 数据集根路径 (包含 train/ 和 val/ 子目录)
        train_size: 每个类别保留的训练样本数，None=使用全部
        val_size: 每个类别保留的验证样本数，None=使用全部
        mlflow_run: MLflow run 对象，如果提供则上传数据分布图
    
    Returns:
        tuple: (train_df, valid_df) - 两个 DataFrame，包含 'filename' 和 'label' 列
    """
    path = Path(data_path).absolute()
    cache_dir = path / ".cache"
    cache_dir.mkdir(exist_ok=True)

    def build_df(root_dir, subset):
        """构建数据集DataFrame，支持缓存和版本检测"""
        cache_file = cache_dir / f"{subset}_df.pkl"
        version_file = cache_dir / f"{subset}_version.txt"
        base_path = root_dir / subset
        
        # 计算数据集版本（基于文件数量和总大小）
        if base_path.exists():
            all_images = list(base_path.rglob("*.*"))
            image_files = [f for f in all_images if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']]
            # 版本 = 文件数量 + 总大小 + 最新修改时间
            total_size = sum(f.stat().st_size for f in image_files)
            latest_mtime = max((f.stat().st_mtime for f in image_files), default=0)
            current_version = f"{len(image_files)}_{total_size}_{latest_mtime}"
        else:
            current_version = "0"
        
        # 检查缓存是否有效
        if cache_file.exists() and version_file.exists():
            cached_version = version_file.read_text().strip()
            if cached_version == current_version:
                if is_main_process():
                    print(f"✓ 使用缓存的 {subset} 数据集 (文件数: {current_version.split('_')[0]})")
                return pd.read_pickle(cache_file)
        
        # 重新构建数据集
        if is_main_process():
            print(f"⚙️  构建 {subset} 数据集（检测到数据变更）...")
        
        records = []
        for img_path in base_path.rglob("*.*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
                rel_path = img_path.relative_to(root_dir)
                class_label = img_path.parent.name
                records.append({
                    "filename": str(rel_path),
                    "label": class_label
                })
        
        df = pd.DataFrame(records)
        
        # 保存缓存和版本
        df.to_pickle(cache_file)
        version_file.write_text(current_version)
        
        if is_main_process():
            print(f"✓ {subset} 数据集构建完成: {len(df)} 张图片")
        
        return df

    train_df = build_df(path, "train")
    valid_df = build_df(path, "val")
    
    # 限制数据集大小（每个类别保留指定数量的样本）
    if train_size and train_size > 0:
        original_train = len(train_df)
        sampled_dfs = []
        for label in train_df['label'].unique():
            label_df = train_df[train_df['label'] == label]
            n_samples = min(train_size, len(label_df))
            sampled_dfs.append(label_df.sample(n=n_samples, random_state=42))
        train_df = pd.concat(sampled_dfs, ignore_index=True)
        if is_main_process():
            print(f"⚠️  训练集已限制（每类{train_size}个样本）: {original_train} → {len(train_df)}")
    
    if val_size and val_size > 0:
        original_val = len(valid_df)
        sampled_dfs = []
        for label in valid_df['label'].unique():
            label_df = valid_df[valid_df['label'] == label]
            n_samples = min(val_size, len(label_df))
            sampled_dfs.append(label_df.sample(n=n_samples, random_state=42))
        valid_df = pd.concat(sampled_dfs, ignore_index=True)
        if is_main_process():
            print(f"⚠️  验证集已限制（每类{val_size}个样本）: {original_val} → {len(valid_df)}")

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    
    # 打印数量
    if is_main_process():
        print(f"  训练集数量: {len(train_df)}")
        print(f"  验证集数量: {len(valid_df)}")
    
    # 生成数据集分布可视化图表并上报到 MLflow (只在主进程)
    if mlflow_run is not None and is_main_process():
        try:
            # 获取类别分布
            train_class_dist = train_df['label'].value_counts().sort_index()
            val_class_dist = valid_df['label'].value_counts().sort_index()
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 获取所有类别（按字母排序）
            all_classes = sorted(set(train_class_dist.index) | set(val_class_dist.index))
            x = np.arange(len(all_classes))
            width = 0.35
            
            # 准备数据
            train_counts = [train_class_dist.get(cls, 0) for cls in all_classes]
            val_counts = [val_class_dist.get(cls, 0) for cls in all_classes]
            
            # 绘制柱状图
            bars1 = ax.bar(x - width/2, train_counts, width, label='Train', alpha=0.8, color='#1f77b4')
            bars2 = ax.bar(x + width/2, val_counts, width, label='Validation', alpha=0.8, color='#ff7f0e')
            
            # 设置标签和标题
            ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
            ax.set_title('Dataset Distribution by Class', fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(all_classes, rotation=45, ha='right')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 在柱状图上添加数值标签
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:  # 只显示非零值
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=8)
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            
            # 添加统计信息文本框
            stats_text = f'Total Train: {len(train_df)}  |  Total Val: {len(valid_df)}  |  Classes: {len(all_classes)}'
            ax.text(0.5, 1.05, stats_text, transform=ax.transAxes,
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            
            # 上传图表到 MLflow
            mlflow.log_figure(fig, "dataset/class_distribution.png")
            plt.close(fig)
            
            print(f"\n📊 数据集分布图已上传到 MLflow")
            print(f"   - 训练集总数: {len(train_df)}")
            print(f"   - 验证集总数: {len(valid_df)}")
            print(f"   - 类别数量: {len(all_classes)}\n")
            
        except Exception as e:
            print(f"⚠️  生成数据集分布图失败: {e}")
            import traceback
            traceback.print_exc()
    
    return train_df, valid_df


def calculate_grad_acc(batch_size, n_gpus, target_batch=256):
    """
    自动计算梯度累积步数，保持有效 batch_size 恒定
    
    Args:
        batch_size: 每个 GPU 的 batch size
        n_gpus: GPU 数量
        target_batch: 目标有效 batch size（默认 256）
    
    Returns:
        int: 梯度累积步数
    
    示例:
        单卡 batch=64  → grad_acc=4  → 有效batch=256
        4卡 batch=64   → grad_acc=1  → 有效batch=256
        单卡 batch=32  → grad_acc=8  → 有效batch=256
    """
    effective_batch_per_step = batch_size * n_gpus
    grad_acc = max(1, target_batch // effective_batch_per_step)
    
    actual_batch = effective_batch_per_step * grad_acc
    
    # 只在主进程输出
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"📊 梯度累积自动配置:")
        print(f"{'='*60}")
        print(f"  输入参数:")
        print(f"    - batch_size (per GPU): {batch_size}")
        print(f"    - n_gpus: {n_gpus}")
        print(f"    - 目标有效 batch: {target_batch}")
        print(f"  计算结果:")
        print(f"    - grad_acc: {grad_acc}")
        print(f"    - 实际有效 batch: {actual_batch}")
        print(f"  内存估算:")
        print(f"    - 梯度内存倍数: ~{grad_acc}x")
        
        # 警告检查
        if actual_batch < target_batch * 0.8:
            print(f"  ⚠️  实际 batch ({actual_batch}) 小于目标的 80%")
        if actual_batch > target_batch * 2:
            print(f"  ⚠️  实际 batch ({actual_batch}) 过大，可能影响收敛")
        
        print(f"{'='*60}\n")
    
    return grad_acc

def train_model(
    data_path,
    model_path='last',
    img_size=320,
    batch_size=256,
    epochs=100,
    lr0=1e-3,
    lrf=0.1,  # 最终学习率比例（相对于初始学习率），默认0.1（10%）
    arch='resnet18',
    wd=1e-3,
    early_stopping=5,
    grad_acc=4,
    load_model=None,
    auto_resume=True,  # 自动加载已存在的best模型继续训练
    task_name='resnet18',
    project_name='ai-classifier',
    train_size=None,  # 每个类别保留的训练样本数，None=使用全部
    val_size=None,    # 每个类别保留的验证样本数，None=使用全部
    device=None,      # 指定GPU设备，例如 'cuda:0', 'cuda:1' 或 None (自动选择)
    scheduler_type='cosine',  # 学习率调度类型: 'cosine', 'cosine_restarts', 'step'
    min_lr=None,  # 最小学习率，None则使用lr*0.01
    distributed=False,  # 是否启用多GPU分布式训练
    models_base_dir='runs',  # 统一的模型保存基础目录
    only_val=False,  # 仅进行验证，不训练
    scale=1.0,  # 图像缩放比例（主要用于分割任务）
    export_onnx=True,  # 训练完成后自动导出ONNX
    mlflow_parent_run_id=None,  # MLflow父run ID（用于嵌套runs）
    skip_mlflow_model_upload=False,  # 跳过模型上传到MLflow（调优模式）
    optimizer='Adam',  # 优化器类型: 'SGD', 'Adam', 'AdamW', 'RMSprop'
    drop_path_rate=0.1,  # DropPath正则化概率（用于ConvNeXt等模型，0=禁用）
):
    
    # 根据架构自动判断任务类型
    is_segmentation = arch.lower().endswith('_seg')
    task_type = 'segmentation' if is_segmentation else 'classification'
    
    # 处理 lr0 参数（兼容YOLO习惯）
    if is_main_process():
        print(f"⚙️  使用 lr0 参数: {lr0}")
    
    # 获取优化器函数
    # 注意：FastAI 需要使用 partial 包装的优化器类（不是实例）
    optimizer_map = {
        'SGD': SGD,
        'Adam': Adam,
        'RAdam': RAdam,
        'RMSprop': RMSProp,
    }
    
    opt_func = optimizer_map.get(optimizer, partial(TorchAdam, betas=(0.9, 0.999), eps=1e-8))
    if is_main_process():
        print(f"🎯 任务类型: {task_type} ({'分割' if is_segmentation else '分类'})")
        print(f"⚙️  优化器: {optimizer}")
        print(f"📊 初始学习率: {lr0}")
    
    # 分割任务的默认参数调整
    if is_segmentation:
        if is_main_process():
            print(f"   分割任务默认 img_size 为 {img_size}")
            print(f"   分割任务默认 batch_size 为 {batch_size}")

    
    # 构建统一的模型保存目录: models_base_dir/project_name/task_name/
    model_save_dir = Path(models_base_dir) / project_name / task_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process():
        print(f"📁 模型保存目录: {model_save_dir.absolute()}")
    
    # 设置GPU设备
    if not distributed:
        if device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1] if ':' in device else device
            print(f"使用指定GPU: {device}")
        else:
            print(f"使用默认GPU配置")
    else:
        # 分布式训练模式
        n_gpus = torch.cuda.device_count()
        if is_main_process():
            print(f"🚀 启用多GPU分布式训练: {n_gpus} GPUs")
            print(f"   进程 Rank: {rank_distrib() if num_distrib() > 1 else 0}/{num_distrib()}")
            
            # 分布式训练时，batch_size是每个GPU的batch大小
            # 总batch_size = batch_size * n_gpus
            print(f"   每GPU batch_size: {batch_size}")
            print(f"   总有效 batch_size: {batch_size * n_gpus}")
    
    # 初始化 MLflow (只在主进程中初始化)
    mlflow_run = None
    if is_main_process():
        try:
            # 如果有父run ID（调优模式），则重用该run
            if mlflow_parent_run_id:
                existing_run = mlflow.active_run()
                if existing_run and existing_run.info.run_id == mlflow_parent_run_id:
                    mlflow_run = existing_run
                    print(f"✅ 使用调优 Trial Run: {mlflow_parent_run_id}")
                else:
                    # 不应该到这里，因为tune.py已经启动了run
                    print(f"⚠️  警告: 父run不是活跃状态，使用现有run")
                    mlflow_run = existing_run
            else:
                # 正常训练模式：检查是否已经有 active run（例如在调优时）
                existing_run = mlflow.active_run()
                if existing_run:
                    print(f"✅ 使用现有 MLflow Run: {existing_run.info.run_id}")
                    mlflow_run = existing_run
                else:
                    mlflow_run = setup_mlflow(project_name, task_name)
                    print(f"✅ MLflow Run ID: {mlflow_run.info.run_id}")
                    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
                    print(f"   Experiment: {project_name}")
                    print(f"   Run Name: {task_name}")
        except Exception as e:
            print(f"⚠️  MLflow 初始化失败: {e}")
            print("   将继续训练但不记录到 MLflow")
    
    # 记录超参数 (只在主进程)
    if mlflow_run is not None:
        try:
            mlflow.log_params({
                'data_path': str(data_path),
                'model_path': model_path,
                'models_base_dir': str(model_save_dir),
                'img_size': img_size,
                'batch_size': batch_size,
                'epochs': epochs,
                'lr0': lr0,
                'lrf': lrf,
                'arch': arch,
                'task_type': task_type,
                'wd': wd,
                'optimizer': optimizer,
                'early_stopping': early_stopping,
                'grad_acc': grad_acc if not is_segmentation else 1,
                'auto_resume': auto_resume,
                'train_size': train_size,
                'val_size': val_size,
                'device': device,
                'scheduler_type': scheduler_type,
                'min_lr': min_lr if min_lr is not None else lr0 * 0.01,
                'distributed': distributed,
                'n_gpus': torch.cuda.device_count() if distributed else 1,
                'scale': scale if is_segmentation else 1.0,
            })
        except Exception as e:
            print(f"⚠️  记录参数到 MLflow 失败: {e}")
    
    path = Path(data_path).absolute()
    
    # 计算合适的 num_workers
    if distributed:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        cpus_per_process = os.cpu_count() // world_size
        num_workers = max(8, max(4, cpus_per_process // 8))
        if is_main_process():
            print(f"💾 DataLoader num_workers: {num_workers} (CPU核心: {os.cpu_count()}, 分布式 {world_size} 进程, 每进程 {cpus_per_process} 核)")
    else:
        gpu_count = max(1, torch.cuda.device_count())
        num_workers = max(8, max(4, os.cpu_count() // gpu_count // 4))
        if is_main_process():
            print(f"💾 DataLoader num_workers: {num_workers} (CPU核心: {os.cpu_count()}, 单机 {gpu_count} GPU)")
    
    if is_main_process():
        print(f"\n🔧 准备 DataLoaders...")
    
    # 根据任务类型创建不同的 DataLoaders
    if is_segmentation:
        # 分割任务：使用 imgs/ 和 masks/ 目录
        dls = get_segmentation_dls(data_dir=data_path, batch_size=batch_size, img_size=img_size, 
                                   scale=scale, num_workers=num_workers, use_disk_cache=True)
        
        if is_main_process():
            print(f"   训练集大小: {len(dls.train_ds)}")
            print(f"   验证集大小: {len(dls.valid_ds)}")
        
        # 分割任务不需要记录类别信息到 MLflow
        
    else:
        # 分类任务：使用 train/ 和 val/ 目录
        train_df, valid_df = load_classification_data(
            data_path=data_path,
            train_size=train_size,
            val_size=val_size,
            mlflow_run=mlflow_run
        )
        
        # 🔧 修复: 打乱验证集，确保多GPU分布式训练时验证准确
        # 问题: 验证集按类别排序 + DistributedSampler按顺序分配
        #      → 不同GPU处理不同类别 → 类别难度差异导致验证loss不准确
        # 解决: 打乱验证集，使所有GPU看到相似的类别分布
        if is_main_process():
            print(f"🔀 打乱验证集以确保多GPU验证准确性...")
        valid_df = valid_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # 合并训练集和验证集，添加 is_valid 列用于分割
        train_df['is_valid'] = False
        valid_df['is_valid'] = True
        combined_df = pd.concat([train_df, valid_df], ignore_index=True)
        
        if is_main_process():
            print(f"   合并后 DataFrame 长度: {len(combined_df)}")
            print(f"   训练集: {(~combined_df['is_valid']).sum()}")
            print(f"   验证集: {combined_df['is_valid'].sum()}")

        # 使用 ImageDataLoaders.from_df 创建 DataLoaders
        dls = ImageDataLoaders.from_df(
            combined_df,
            path=path,
            valid_col='is_valid',
            fn_col='filename',
            label_col='label',
            num_workers=num_workers,
            item_tfms=Resize(img_size, method='bicubic'),  # 使用 bicubic 插值（Timm ConvNeXt 推荐）
            bs=batch_size,
            batch_tfms=aug_transforms(flip_vert=False, max_rotate=5, max_zoom=1.05, max_lighting=0.1)
        )
        
        if is_main_process():
            print(f"   分类: {dls.vocab}")
            print(f"   train_df 长度: {len(train_df)}")
            print(f"   dls.train.dataset 长度: {len(dls.train.dataset)}")
            print(f"   是否相等: {len(train_df) == len(dls.train.dataset)}") 

        # 记录类别信息 (只在主进程)
        if mlflow_run is not None:
            try:
                mlflow.log_metric('dataset/num_classes', len(dls.vocab))
                import json
                class_names_path = model_save_dir / 'class_names.json'
                with open(class_names_path, 'w') as f:
                    json.dump(list(dls.vocab), f)
                print(f"   保存类别名称到: {class_names_path}")
                mlflow.log_artifact(str(class_names_path))
            except Exception as e:
                print(f"⚠️  记录类别信息到 MLflow 失败: {e}")
    
    # 自动加载之前的模型（如果存在且启用了auto_resume）
    resume_from_epoch = 0
    resume_best_metric = None
    
    # 如果没有明确指定 load_model，且启用了 auto_resume，尝试自动加载
    if load_model is None and auto_resume:
        # 检查是否存在之前的 best 模型
        auto_load_path = model_save_dir / 'best.pth'
        if auto_load_path.exists():
            if is_main_process():
                print(f"🔍 发现已存在的模型: {auto_load_path}")
                print(f"   自动加载以继续训练...")
            load_model = str(auto_load_path)
    
    # 从checkpoint恢复epoch信息和best_metric (只在主进程打印)
    if load_model is not None:
        if is_main_process():
            print(f"\n{'='*80}")
            print(f"📦 从 checkpoint 加载模型")
            print(f"{'='*80}")
            print(f"模型路径: {load_model}")
        
        # 先加载状态字典以获取epoch信息
        state_dict = torch.load(load_model, map_location='cpu')
        
        if is_main_process():
            print(f"\n📋 Checkpoint 信息:")
            
        # 检查是否包含epoch信息
        if isinstance(state_dict, dict) and 'epoch' in state_dict:
            # +1 因为保存的是完成的epoch，下一次训练从下一个epoch开始
            resume_from_epoch = state_dict['epoch'] + 1
            if is_main_process():
                print(f"  - 已完成的 epoch: {state_dict['epoch']}")
                print(f"  - 下次训练起始 epoch: {resume_from_epoch}")
        else:
            if is_main_process():
                print(f"  - Epoch 信息: ⚠️  未找到")
                print(f"  - 将从 epoch 0 开始计算学习率")
        
        # 检查是否包含loss信息
        if isinstance(state_dict, dict) and 'loss' in state_dict:
            resume_best_metric = state_dict['loss']
            if is_main_process():
                print(f"  - 当前 valid_loss: {resume_best_metric:.6f}")
        else:
            if is_main_process():
                print(f"  - 最佳指标: ⚠️  未找到")
        
        # 检查其他信息
        if is_main_process():
            if isinstance(state_dict, dict):
                if 'img_size' in state_dict:
                    print(f"  - 图像尺寸: {state_dict['img_size']}")
                if 'arch' in state_dict:
                    print(f"  - 模型架构: {state_dict['arch']}")
                if 'opt' in state_dict:
                    print(f"  - 优化器状态: ✅ 已保存")
                else:
                    print(f"  - 优化器状态: ⚠️  未找到")
            
            print(f"{'='*80}\n")
    else:
        if is_main_process():
            print("🆕  从头开始训练新模型")
    
    # 创建learner（在加载模型之前，因为需要resume_from_epoch信息）
    # 构建callbacks列表
    callbacks = [
        YOLOv11LRScheduler(epochs=epochs, lr0=lr0, lrf=lrf, warmup_epochs=3, 
                          resume_from_epoch=resume_from_epoch, min_lr=min_lr, 
                          scheduler_type=scheduler_type),
        ResumeEpochCallback(resume_from_epoch=resume_from_epoch),
    ]
    
    # 分割任务不使用梯度累积
    if not is_segmentation:
        # 如果 grad_acc <= 0，自动计算
        if grad_acc <= 0:
            n_gpus = torch.cuda.device_count() if distributed else 1
            grad_acc = calculate_grad_acc(batch_size, n_gpus, target_batch=256)
            
            # 记录自动计算的 grad_acc 到 MLflow
            if mlflow_run is not None and is_main_process():
                try:
                    effective_batch = batch_size * n_gpus * grad_acc
                    mlflow.log_params({
                        'grad_acc_auto': grad_acc,
                        'effective_batch_size': effective_batch
                    })
                except Exception as e:
                    print(f"⚠️  记录 grad_acc 到 MLflow 失败: {e}")
        
        # 🔧 修复：多GPU训练时禁用梯度累积，避免loss双重缩放问题
        # 问题：FastAI的GradientAccumulation会将loss除以n_acc，但DDP已经平均了loss
        # 结果：loss被错误地双重缩放，导致梯度过小，模型不收敛
        if distributed and grad_acc > 1:
            if is_main_process():
                print(f"\n{'='*80}")
                print(f"⚠️  检测到多GPU分布式训练 + 梯度累积，这会导致loss计算冲突！")
                print(f"{'='*80}")
                print(f"问题说明：")
                print(f"  - FastAI的GradientAccumulation会缩放loss: loss /= {grad_acc}")
                print(f"  - 但DDP已经自动在所有GPU间平均了loss")
                print(f"  - 双重缩放导致梯度过小，模型几乎不学习")
                print(f"解决方案：")
                print(f"  - 原grad_acc={grad_acc} → 强制设为1（禁用梯度累积）")
                print(f"  - 有效batch_size={batch_size} x {n_gpus} GPUs = {batch_size * n_gpus}")
                print(f"  - 如需更大batch，请增大 --batch_size 参数")
                print(f"{'='*80}\n")
            grad_acc = 1
        
        # 只在grad_acc > 1时添加梯度累积回调
        if grad_acc > 1:
            callbacks.insert(0, GradientAccumulation(n_acc=grad_acc))
    
    callbacks.append(EarlyStoppingWithEvalCallback(
        monitor='valid_loss', 
        patience=early_stopping, 
        resume_best_metric=resume_best_metric
    ))
    
    # # 分布式训练时添加验证诊断回调（帮助发现验证loss问题）
    # if distributed:
    #     callbacks.append(DistributedValidationDiagnosticCallback(verbose=True))
    #     if is_main_process():
    #         print("📊 已启用分布式验证诊断回调，将在每个epoch后报告GPU间loss差异")
    
    # 只在主进程添加 MLflow 和模型保存回调
    if is_main_process():
        callbacks.extend([
            SaveModelWithEpochCallback(monitor='valid_loss', fname='best', last_fname=model_path,
                                      with_opt=True, resume_from_epoch=resume_from_epoch,
                                      img_size=img_size, arch=arch, 
                                      resume_best_metric=resume_best_metric,
                                      save_dir=model_save_dir, save_last=True,
                                      upload_to_mlflow=(mlflow_run is not None and not skip_mlflow_model_upload)),
            MLflowMetricsCallback(resume_from_epoch=resume_from_epoch) if mlflow_run is not None else None
        ])
        callbacks = [cb for cb in callbacks if cb is not None]
    
    # 根据任务类型创建 Learner
    if is_segmentation:
        # 分割任务
        if is_main_process():
            print(f"✅ 使用分割模型: {arch}")
        
        # 分割模型必须是自定义模型
        if not (CUSTOM_MODELS_AVAILABLE and is_custom_model(arch)):
            raise ValueError(f"分割模型 '{arch}' 未找到，请确保使用 'unet_seg' 等分割模型")
        
        model = get_model(arch, n_classes=1)  # 二分类分割
        
        # 输出模型结构
        if is_main_process():
            input_size = (1, 3, img_size, img_size)
            print_model_structure(model, model_name=arch, input_size=input_size)
        
        learn = Learner(
            dls,
            model,
            loss_func=CombinedLoss(bce_weight=0.5, dice_weight=0.5),
            opt_func=opt_func,
            metrics=[DiceMetric()],
            wd=wd,
            cbs=callbacks
        )
        
        # 显示 GPU 信息
        if is_main_process():
            if torch.cuda.is_available():
                print(f"📍 训练将使用 GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"⚠️  警告: 将使用 CPU 训练")
        
    else:
        # 分类任务
        if CUSTOM_MODELS_AVAILABLE and is_custom_model(arch):
            # 使用自定义分类模型
            if is_main_process():
                print(f"✅ 使用自定义模型: {arch}")
            
            custom_model = get_model(arch, n_classes=len(dls.vocab))

            # 输出模型结构
            if is_main_process():
                input_size = (1, 3, img_size, img_size)
                print_model_structure(custom_model, model_name=arch, input_size=input_size)
             
            learn = Learner(
                dls,
                custom_model,
                opt_func=opt_func,
                metrics=[
                    accuracy,
                    error_rate,
                    Precision(average='weighted'),
                    Recall(average='weighted'),
                    F1Score(average='weighted'),
                ],
                wd=wd,
                cbs=callbacks
            )
        else:
            # 使用timm模型（通过vision_learner包装）
            try:
                import timm
                
                # 检查是否为timm模型
                if arch not in timm.list_models():
                    print(f"\n❌ 错误: '{arch}' 不是有效的模型")
                    print(f"\n💡 提示:")
                    print(f"   - 使用 --list-models 查看所有可用模型")
                    print(f"   - Timm模型: resnet18, efficientnet_b0, vit_base_patch16_224 等")
                    print(f"   - 自定义模型: yolov11s_cls, unet_seg 等")
                    sys.exit(1)
                
                if is_main_process():
                    print(f"✅ 使用Timm模型（vision_learner包装）: {arch}")
                
                # 检测是否为 ConvNeXt 模型，自动提示 DropPath 配置
                model_kwargs = {}
                if 'convnext' in arch.lower() and drop_path_rate > 0:
                    model_kwargs['drop_path_rate'] = drop_path_rate
                    if is_main_process():
                        print(f"   ✓ 启用 DropPath 正则化: drop_path_rate={drop_path_rate}")
                        print(f"   推荐值: Tiny=0.1, Small=0.2, Base=0.3, Large=0.4")
                elif 'convnext' in arch.lower() and drop_path_rate == 0:
                    if is_main_process():
                        print(f"   ⚠️  警告: ConvNeXt 模型未启用 DropPath (drop_path_rate=0)")
                        print(f"   这可能导致训练效果变差，建议设置 --drop_path_rate 0.1")
                
                # 使用vision_learner创建模型（FastAI会自动处理timm模型）
                # 强制不使用预训练权重，并传递 drop_path_rate 等参数
                learn = vision_learner(
                    dls, 
                    arch=arch,
                    pretrained=False,
                    metrics=[
                        accuracy,
                        error_rate,
                        Precision(average='weighted'),  
                        Recall(average='weighted'),    
                        F1Score(average='weighted'),   
                    ],
                    opt_func=opt_func,
                    cbs=callbacks,
                    **model_kwargs  # 传递额外参数（如 drop_path_rate）
                )
                
                # 输出模型结构
                if is_main_process():
                    input_size = (1, 3, img_size, img_size)
                    print_model_structure(learn.model, model_name=arch, input_size=input_size)
                
            except ImportError:
                print(f"\n❌ 错误: 未安装timm库")
                print(f"\n安装命令:")
                print(f"   pip install timm")
                sys.exit(1)

    # 加载模型权重和优化器状态 (只在主进程打印详细信息)
    optimizer_state_to_load = None  # 保存优化器状态，稍后加载
    if load_model is not None:
        # 获取模型设备
        device = next(learn.model.parameters()).device
        
        # 重新加载到正确的设备
        state_dict = torch.load(load_model, map_location=device)
        
        # 检查是否是 FastAI 保存的格式（包含 'model' 和 'opt' 键）
        if isinstance(state_dict, dict) and 'model' in state_dict:
            if is_main_process():
                print("加载模型权重...")
            
            # 加载模型权重
            # FastAI会自动处理DDP模型的'module.'前缀问题
            model_state = state_dict['model']
            learn.model.load_state_dict(model_state)
            
            # 保存优化器状态，稍后在优化器初始化后加载
            if 'opt' in state_dict and state_dict['opt'] is not None:
                optimizer_state_to_load = state_dict['opt']
                if is_main_process():
                    print("💾 优化器状态已准备，将在训练开始后恢复")
                    
                    # 显示保存的优化器参数
                    try:
                        if 'param_groups' in optimizer_state_to_load and len(optimizer_state_to_load['param_groups']) > 0:
                            saved_pg = optimizer_state_to_load['param_groups'][0]
                            print("📋 checkpoint中保存的优化器参数:")
                            print(f"   - 学习率 (lr): {saved_pg.get('lr', 'N/A')}")
                    except Exception as e:
                        print(f"   无法显示保存的优化器参数: {e}")
            else:
                if is_main_process():
                    print("⚠️  checkpoint中没有优化器状态")
            
            # 显示学习率调度信息
            if is_main_process():
                print(f"📊 学习率调度信息:")
                print(f"   - 从epoch {resume_from_epoch}继续训练")
                if resume_from_epoch < 3:
                    print(f"   - 当前仍在热身阶段 (warmup_epochs=3)")
                else:
                    print(f"   - 已过热身阶段，使用余弦退火调度")
        else:
            # 直接加载权重（纯模型权重，无优化器状态）
            learn.model.load_state_dict(state_dict)
            if is_main_process():
                print("⚠️  加载纯模型权重（无优化器状态和epoch信息）")
        
        if is_main_process():
            print("✅ 模型权重加载完成")
    
    # 如果有优化器状态需要加载，添加对应的callback
    # 支持覆盖超参数（例如，如果命令行指定了新的 wd）
    if optimizer_state_to_load is not None:
        # 准备要覆盖的超参数（如果命令行指定了不同的值）
        override_hypers = {}
        
        # 检查 checkpoint 中的 wd 和当前命令行指定的 wd 是否不同
        if 'param_groups' in optimizer_state_to_load and len(optimizer_state_to_load['param_groups']) > 0:
            checkpoint_wd = optimizer_state_to_load['param_groups'][0].get('wd', 
                           optimizer_state_to_load['param_groups'][0].get('weight_decay', None))
            
            if checkpoint_wd is not None and abs(checkpoint_wd - wd) > 1e-6:
                if is_main_process():
                    print(f"\n⚠️  检测到权重衰减不一致:")
                    print(f"   Checkpoint 中: {checkpoint_wd}")
                    print(f"   命令行指定: {wd}")
                    print(f"   将使用命令行指定的值: {wd}")
                override_hypers['wd'] = wd
        
        learn.add_cb(LoadOptimizerStateCallback(optimizer_state_to_load, override_hypers=override_hypers))

    if not only_val:
        try:
            # 配置DDP参数以支持有未使用参数的模型（如ConvNeXt等）
            # 注意：ConvNeXt等模型的深度可分离卷积会产生非连续梯度布局，
            # 确保DDP使用拷贝模式而非视图模式
            gradient_as_bucket_view=False 
            if distributed:
                ddp_kwargs = DistributedDataParallelKwargs(
                    find_unused_parameters=True,
                    gradient_as_bucket_view=False  # 避免步幅不匹配（ConvNeXt深度卷积）
                )
                if is_main_process():
                    print("🔧 已配置DDP: find_unused_parameters=True, gradient_as_bucket_view=False")
            else:
                ddp_kwargs = None
            
            # 传递kwargs_handlers到distrib_ctx
            ctx_kwargs = {'kwargs_handlers': [ddp_kwargs]} if ddp_kwargs else {}
            
            with learn.distrib_ctx(**ctx_kwargs):
                learn.fit(epochs, lr=lr0, wd=wd)
            
        except CancelFitException:
            # 早停触发，这是正常的，不是错误
            if is_main_process():
                print("\n✅ 训练因早停而结束")
        
        except Exception as e:
            print("错误:", file=sys.stderr)
            traceback.print_exc()
            raise
        
        finally:
            # 训练完成后的清理
            # FastAI的to_parallel()会自动管理DDP生命周期，无需手动清理
            if is_main_process():
                print("\n✅ 训练完成")
            
            # 非主进程在分布式模式下退出（不参与评估）
            if distributed and not is_main_process():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
    
    # 🔧 关键修复：在评估前先清理分布式环境
    # 问题：评估时get_preds()在分布式环境下会触发进程同步，导致OOM和SIGABRT
    # 解决：主进程在评估前清理分布式环境，改为单GPU模式评估
    if distributed and is_main_process():
        print("\n🔧 清理分布式环境（准备单GPU评估）...")
        
        # 1. 移除learner中的分布式callbacks
        from fastai.distributed import DistributedTrainer, GatherPredsCallback
        callbacks_to_remove = []
        for cb in learn.cbs:
            if isinstance(cb, (DistributedTrainer, GatherPredsCallback)):
                callbacks_to_remove.append(cb)
                print(f"   - 移除callback: {cb.__class__.__name__}")
        
        for cb in callbacks_to_remove:
            learn.remove_cb(cb)
        
        # 2. 如果模型被DDP包装，提取原始模型
        if hasattr(learn.model, 'module'):
            print(f"   - 提取DDP包装的原始模型")
            learn.model = learn.model.module
        
        # 3. 清理PyTorch分布式环境
        if torch.distributed.is_initialized():
            print(f"   - 销毁分布式进程组")
            torch.distributed.destroy_process_group()
        
        # 4. 清理环境变量
        distributed_env_vars = [
            'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 
            'MASTER_ADDR', 'MASTER_PORT',
            'TORCH_DISTRIBUTED_DEBUG'
        ]
        for var in distributed_env_vars:
            if var in os.environ:
                del os.environ[var]
        
        print(f"   ✅ 分布式环境已清理，现在以单GPU模式继续")
    
    # 以下代码只有主进程会执行（或单GPU训练）
    if is_main_process():
        # 注意：best指标已在训练过程中实时上报到MLflow（每次更新best时）
        # 无需在训练结束后再次上报
        
        # 上传模型到 MLflow
        if mlflow_run is not None and not skip_mlflow_model_upload:
            try:
                
                # best 模型已在训练过程中实时上传，这里只需要调用评估
                best_model_path = model_save_dir / 'best.pth'
                if best_model_path.exists():
                    print(f"ℹ️  最佳模型已在训练中上传到 MLflow: best.pth")
                    
                    # 分割任务不需要最终评估
                    if not is_segmentation:
                        # 调用独立的评估模块（复用训练数据）
                        call_evaluation_script(
                            learn=learn,
                            model_path=best_model_path,
                            mlflow_run_id=mlflow_run.info.run_id,
                            project_name=project_name
                        )
                    else:
                        print(f"ℹ️  分割任务跳过最终评估")
                    
                    # 导出ONNX模型
                    if export_onnx:
                        try:
                            print(f"\n{'='*80}")
                            print(f"📦 导出ONNX模型")
                            print(f"{'='*80}")
                            
                            from export_onnx import export_to_onnx
                            
                            onnx_path = model_save_dir / 'best.onnx'
                            export_to_onnx(
                                model_path=str(best_model_path),
                                arch=arch,
                                output_path=str(onnx_path),
                                img_size=img_size,
                                device='cpu',  # ONNX导出在CPU上进行
                                data_path=str(data_path),
                                classes=None  # 从数据路径自动获取
                            )
                            
                            # 上传ONNX模型到MLflow
                            if mlflow_run is not None and onnx_path.exists():
                                try:
                                    mlflow.log_artifact(str(onnx_path), artifact_path="models")
                                    print(f"✅ ONNX模型已上传到 MLflow")
                                except Exception as e:
                                    print(f"⚠️  上传ONNX到 MLflow 失败: {e}")
                            
                            print(f"{'='*80}\n")
                        except Exception as e:
                            print(f"⚠️  导出ONNX失败: {e}")
                            import traceback as tb
                            tb.print_exc()
                    else:
                        print(f"ℹ️  跳过ONNX导出（--no-export-onnx）")
                else:
                    print(f"⚠️ 未找到best模型，跳过评估: {best_model_path}")
            except Exception as e:
                print(f"⚠️  上传模型到 MLflow 失败: {e}")
        elif skip_mlflow_model_upload:
            print(f"ℹ️  调优模式：跳过模型上传到 MLflow")
        
        # 完成 MLflow run
        if mlflow_run is not None:
            mlflow.end_run()
        
        print("\n✅ 训练完成!")




def list_available_models(series_filter=None):
    """
    列出所有可用的模型（Timm、自定义模型）
    
    Args:
        series_filter: 可选，指定模型系列名称（如 'resnet', 'efficientnet'等）
                      如果指定，则显示该系列下的所有模型；否则显示所有系列摘要
    """
    print("\n" + "="*80)
    if series_filter:
        print(f"📋 模型系列: {series_filter}")
    else:
        print("📋 可用模型目录")
    print("="*80)
    
    # 1. Timm 模型（动态获取）
    timm_series = {}
    try:
        import timm
        timm_models = timm.list_models()
        
        # 自动分类所有timm模型（按前缀）
        from collections import defaultdict
        timm_by_prefix = defaultdict(list)
        for model in sorted(timm_models):
            # 提取前缀（第一个下划线或数字前的部分）
            import re
            match = re.match(r'^([a-z]+)', model)
            if match:
                prefix = match.group(1)
                timm_by_prefix[prefix].append(model)
        
        timm_series = dict(timm_by_prefix)
    except ImportError:
        pass
    
    # 如果指定了系列过滤器，显示该系列的所有模型
    if series_filter:
        series_lower = series_filter.lower()
        found = False
        
        # 搜索 Timm 模型
        if series_lower in timm_series:
            models = timm_series[series_lower]
            print(f"\n1️⃣  Timm - {series_filter} ({len(models)} 个模型):")
            for i, model in enumerate(models, 1):
                print(f"   {i:3d}. {model}")
            found = True
        
        # 搜索自定义模型
        if CUSTOM_MODELS_AVAILABLE:
            custom_models = list_models()
            matched = [m for m in custom_models if m.lower().startswith(series_lower)]
            if matched:
                print(f"\n2️⃣  自定义模型 - {series_filter} ({len(matched)} 个模型):")
                for i, model in enumerate(matched, 1):
                    model_type = "分割" if model.endswith('_seg') else "分类"
                    print(f"   {i:2d}. {model:30s} [{model_type}]")
                found = True
        
        if not found:
            print(f"\n⚠️  未找到系列 '{series_filter}' 的模型")
            print(f"\n💡 查看所有系列: python train.py --list-models")
        
    else:
        # 显示所有系列摘要
        if timm_series:
            print(f"\n1️⃣  Timm 模型库 (共 {len(timm_series)} 个系列):")
            # 只显示模型数量 >= 3 的主要系列
            major_series = {k: v for k, v in timm_series.items() if len(v) >= 3}
            for series_name in sorted(major_series.keys()):
                count = len(major_series[series_name])
                print(f"   • {series_name:15s} ({count:3d} 个模型)")
            
            minor_count = len(timm_series) - len(major_series)
            if minor_count > 0:
                print(f"   ... 以及 {minor_count} 个其他小系列")
        else:
            print(f"\n1️⃣  Timm 模型库: ⚠️  未安装 (pip install timm)")
        
        if CUSTOM_MODELS_AVAILABLE:
            custom_models = list_models()
            classification_models = [m for m in custom_models if not m.endswith('_seg')]
            segmentation_models = [m for m in custom_models if m.endswith('_seg')]
            
            print(f"\n2️⃣  自定义模型:")
            if classification_models:
                print(f"   • 分类模型 ({len(classification_models)} 个):")
                for i, model in enumerate(classification_models, 1):
                    print(f"     {i}. {model}")
            if segmentation_models:
                print(f"   • 分割模型 ({len(segmentation_models)} 个):")
                for i, model in enumerate(segmentation_models, 1):
                    print(f"     {i}. {model}")
        else:
            print("\n2️⃣  自定义模型: ⚠️  未加载")
        
        print("\n" + "="*80)
        print("\n💡 查看特定系列的所有模型:")
        print("   python train.py --list-models resnet")
        print("   python train.py --list-models efficientnet")
        print("   python train.py --list-models vit")
        print("   python train.py --list-models yolo  # 查看自定义YOLO模型")
    
    print("\n" + "="*80)
    print("\n💡 使用方法:")
    print("   Timm模型:   python train.py --arch resnet18 --data_path <数据集路径>")
    print("   自定义模型: python train.py --arch yolov11s_cls --data_path <数据集路径>")
    print("\n📁 数据集格式:")
    print("   分类: data_path/train/class1/, data_path/val/class1/...")
    print("   分割: data_path/imgs/train/, data_path/masks/train/...")
    print()

def print_structure_command(arch, img_size=224, show_shape=True):
    """
    专门用于打印模型结构的命令函数
    
    Args:
        arch: 模型架构名称
        img_size: 输入图像尺寸
        show_shape: 是否显示shape信息
    """
    print("\n" + "="*80)
    print("📊 打印模型结构")
    print("="*80)
    print(f"\n模型: {arch}")
    print(f"输入尺寸: {img_size}x{img_size}")
    print(f"显示Shape: {'是' if show_shape else '否'}")
    
    try:
        # 检查是否是自定义模型
        if CUSTOM_MODELS_AVAILABLE and is_custom_model(arch):
            print(f"\n✅ 使用自定义模型: {arch}")
            # 使用默认类别数（用于显示结构）
            model = get_model(arch, n_classes=10)
        else:
            # 使用timm模型
            import timm
            if arch in timm.list_models():
                print(f"\n✅ 使用Timm模型: {arch}")
                model = timm.create_model(arch, pretrained=False, num_classes=10)
            else:
                print(f"\n❌ 未找到模型: {arch}")
                print("\n💡 使用 --list-models 查看所有可用模型")
                return
        
        # 打印结构
        input_size = (1, 3, img_size, img_size)
        print_model_structure(model, model_name=arch, input_size=input_size, show_shape=show_shape)
        
    except Exception as e:
        print(f"\n❌ 打印模型结构失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💡 提示:")
    print("   - 使用 --arch 指定不同的模型")
    print("   - 使用 --img_size 指定不同的输入尺寸")
    print("   - 使用 --no-shape 关闭shape显示（加快速度）")
    print()


def main():
    parser = argparse.ArgumentParser(description='训练医学图像分类模型')
    
    # 数据和模型路径
    parser.add_argument('--data_path', type=str, default='/mnt/ssd/dataset', help='数据集路径')
    parser.add_argument('--model_path', type=str, default='last', help='保存模型的文件名（不含路径）')
    parser.add_argument('--load_model', type=str, default=None, help='加载已有的模型继续训练（完整路径）')
    parser.add_argument('--auto_resume', action='store_true', default=True, 
                       help='自动加载已存在的best模型继续训练（默认开启）')
    parser.add_argument('--no-auto-resume', dest='auto_resume', action='store_false', 
                       help='禁用自动恢复，强制从头训练')
    parser.add_argument('--models_base_dir', type=str, default='runs', 
                       help='统一的模型保存基础目录，实际保存路径为: models_base_dir/project_name/task_name/')
    
    # 模型和训练参数
    parser.add_argument('--img_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.1, help='最终学习率比例（相对于lr），推荐0.1-0.2')
    parser.add_argument('--arch', type=str, default='resnet18', help='模型架构')
    parser.add_argument('--wd', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--early_stopping', type=int, default=100, help='早停轮数')
    parser.add_argument('--grad_acc', type=int, default=0, 
                       help='梯度累积步数 (0=自动计算, >0=手动指定, 默认0自动)')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                       help='DropPath正则化概率（用于ConvNeXt等模型，0=禁用，推荐ConvNeXt-Tiny:0.1, Small:0.2, Base:0.3）')
    
    # 学习率调度参数
    parser.add_argument('--scheduler_type', type=str, default='cosine', 
                       choices=['cosine', 'cosine_restarts', 'step'],
                       help='学习率调度类型: cosine=标准余弦, cosine_restarts=周期重启, step=分段衰减')
    parser.add_argument('--min_lr', type=float, default=None, help='最小学习率，None则使用lr*0.01')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='Adam',
                       choices=['SGD', 'Adam', 'AdamW', 'RMSprop'],
                       help='优化器类型（默认: Adam）')
    
    # 分割任务专用参数
    parser.add_argument('--scale', type=float, default=1.0, help='图像缩放比例（分割任务，用于节省显存）')
    
    # MLflow参数
    parser.add_argument('--task_name', type=str, default='Image Classification', help='MLflow运行名称')
    parser.add_argument('--project_name', type=str, default='ai-classifier', help='MLflow实验名称')
    
    # 数据集大小限制（用于快速验证）
    parser.add_argument('--train_size', type=int, default=None, help='每个类别保留的训练样本数，None=使用全部')
    parser.add_argument('--val_size', type=int, default=None, help='每个类别保留的验证样本数，None=使用全部')
    
    # GPU设备选择
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备，例如 "0", "1" 或 "cuda:0", None=使用默认')
    
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true', help='启用多GPU分布式训练')

    parser.add_argument('--only_val', action='store_true', help='不训练，只使用 best 模型评估')
    
    # ONNX导出
    parser.add_argument('--export_onnx', action='store_true', default=True,
                       help='训练完成后自动导出ONNX模型（默认开启）')
    parser.add_argument('--no-export-onnx', dest='export_onnx', action='store_false',
                       help='禁用自动导出ONNX')
    
    # 列出可用模型
    parser.add_argument('--list-models', type=str, nargs='?', const='', default=None,
                       metavar='SERIES',
                       help='列出可用的模型。不带参数显示所有系列，带参数显示指定系列的所有模型（如: --list-models resnet）')
    
    # 仅显示模型结构
    parser.add_argument('--show-model', action='store_true',
                       help='仅显示模型结构，不进行训练（配合--arch使用）')
    
    # 打印模型结构（新命令）
    parser.add_argument('--print-structure', action='store_true',
                       help='打印模型详细结构（包括每层的shape信息，配合--arch使用）')
    parser.add_argument('--no-shape', action='store_true',
                       help='打印结构时不显示shape信息（加快速度）')
    
    args = parser.parse_args()

    # 如果用户请求列出模型，显示后退出
    if args.list_models is not None:
        # args.list_models 为空字符串表示不带参数，显示所有系列
        # args.list_models 为具体值表示指定系列
        series_filter = args.list_models if args.list_models else None
        list_available_models(series_filter)
        return

    # 如果用户请求打印结构
    if args.print_structure:
        print_structure_command(
            arch=args.arch,
            img_size=args.img_size,
            show_shape=not args.no_shape
        )
        return

    if is_main_process(): 
        # 显示模型保存路径信息
        print(f"\n{'='*80}")
        print(f"模型保存配置:")
        print(f"  基础目录: {args.models_base_dir}")
        print(f"  项目名称: {args.project_name}")
        print(f"  任务名称: {args.task_name}")
        print(f"  实际路径: {Path(args.models_base_dir).absolute() / args.project_name / args.task_name}")
        print(f"{'='*80}\n")
    
    # 统一的参数字典，避免重复代码
    train_params = {
        'data_path': args.data_path,
        'model_path': args.model_path,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'arch': args.arch,
        'wd': args.wd,
        'early_stopping': args.early_stopping,
        'grad_acc': args.grad_acc,
        'load_model': args.load_model,
        'auto_resume': args.auto_resume,
        'task_name': args.task_name,
        'project_name': args.project_name,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'device': args.device,
        'scheduler_type': args.scheduler_type,
        'min_lr': args.min_lr,
        'distributed': args.distributed,
        'models_base_dir': args.models_base_dir,
        'only_val': args.only_val,
        'scale': args.scale,
        'export_onnx': args.export_onnx,
        'optimizer': args.optimizer,
        'drop_path_rate': args.drop_path_rate,
    }
    
    # FastAI 分布式训练：在主函数中初始化环境，然后正常调用训练
    if args.distributed:
        setup_distrib()
        try:
            train_model(**train_params)
        finally:
            teardown_distrib()
    else:
        train_model(**train_params)

if __name__ == '__main__':
    main()
