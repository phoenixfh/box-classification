"""
训练回调函数

包含学习率调度、早停、模型保存等训练回调。
"""

import os
import math
import numpy as np
import torch
import mlflow
from pathlib import Path
from fastai.callback.core import Callback, CancelFitException
from fastai.callback.tracker import TrackerCallback, Recorder

from .data_loading import is_main_process


class YOLOv11LRScheduler(Callback):
    """
    改进的学习率调度器：
    - 热身阶段：线性增加
    - 余弦退火：保持较高的最终学习率（lrf建议0.1-0.2）
    - 支持最小学习率限制，避免后期学习率过小
    """
    def __init__(self, epochs=100, lr0=0.01, lrf=0.1, warmup_epochs=3, warmup_momentum=0.8, 
                 resume_from_epoch=0, min_lr=None, scheduler_type='cosine'):
        self.lr0 = lr0
        self.lrf = lrf
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        self.final_lr = lr0 * lrf
        self.total_epochs = epochs + resume_from_epoch
        self.resume_from_epoch = resume_from_epoch
        self.scheduler_type = scheduler_type  # 'cosine', 'cosine_restarts', 'step'
        
        # 设置最小学习率（默认为初始学习率的1%）
        self.min_lr = min_lr if min_lr is not None else lr0 * 0.01
        # 确保final_lr不低于min_lr
        self.final_lr = max(self.final_lr, self.min_lr)
        
    def before_fit(self):
        self.optimizer = self.learn.opt
        
    def before_epoch(self):
        current_epoch = self.learn.epoch + self.resume_from_epoch
        
        # 热身阶段
        if current_epoch < self.warmup_epochs:
            lr = self.lr0 * (0.1 + 0.9 * (current_epoch / self.warmup_epochs))
            momentum = 0.9 - (0.9 - 0.85) * (1 - current_epoch / self.warmup_epochs)
        else:
            # 主训练阶段：根据scheduler_type选择不同的调度策略
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            
            if self.scheduler_type == 'cosine':
                # 标准余弦退火（改进：更高的最终学习率）
                lr = self.final_lr + 0.5 * (self.lr0 - self.final_lr) * (1 + math.cos(math.pi * progress))
                
            elif self.scheduler_type == 'cosine_restarts':
                # 余弦退火 + 周期性重启（每1/3周期重启一次）
                restart_period = (self.total_epochs - self.warmup_epochs) / 3
                cycle_progress = (current_epoch - self.warmup_epochs) % restart_period / restart_period
                lr = self.final_lr + 0.5 * (self.lr0 - self.final_lr) * (1 + math.cos(math.pi * cycle_progress))
                
            elif self.scheduler_type == 'step':
                # 分段余弦衰减：前70%余弦衰减，后30%保持较高学习率
                if progress < 0.7:
                    # 前70%使用余弦衰减到final_lr
                    local_progress = progress / 0.7
                    lr = self.final_lr + 0.5 * (self.lr0 - self.final_lr) * (1 + math.cos(math.pi * local_progress))
                else:
                    # 后30%保持final_lr（比标准余弦退火高）
                    lr = self.final_lr
            else:
                # 默认使用标准余弦退火
                lr = self.final_lr + 0.5 * (self.lr0 - self.final_lr) * (1 + math.cos(math.pi * progress))
            
            # 应用最小学习率限制
            lr = max(lr, self.min_lr)
            momentum = 0.9
        
        # 使用 fastai 的 set_hypers 方法更新学习率和动量
        self.optimizer.set_hypers(lr=lr, mom=momentum)


class LoadOptimizerStateCallback(Callback):
    """在训练开始前加载优化器状态"""
    def __init__(self, optimizer_state, override_hypers=None):
        """
        Args:
            optimizer_state: 要加载的优化器状态字典
            override_hypers: 可选，覆盖恢复的超参数的字典
                            例如: {'wd': 0.001, 'lr': 0.01}
        """
        self.optimizer_state = optimizer_state
        self.override_hypers = override_hypers or {}
        self.loaded = False
    
    def before_fit(self):
        """在fit开始前尝试加载优化器状态"""
        if not self.loaded and self.optimizer_state is not None:
            # 此时优化器可能还未初始化，先等待
            pass
    
    def before_train(self):
        """在训练循环开始前加载优化器状态（此时优化器已初始化）"""
        if not self.loaded and self.optimizer_state is not None:
            try:
                # 确保优化器已经初始化
                if self.learn.opt is None:
                    print("⚠️  优化器尚未初始化，等待下一次尝试...")
                    return
                
                # 获取模型所在的设备
                device = next(self.learn.model.parameters()).device
                
                # 将优化器状态移动到正确的设备
                optimizer_state_on_device = self._move_optimizer_state_to_device(
                    self.optimizer_state, device
                )
                
                # 加载优化器状态
                self.learn.opt.load_state_dict(optimizer_state_on_device)
                
                # 如果有覆盖的超参数，应用它们
                if self.override_hypers:
                    if is_main_process():
                        print(f"\n🔧 覆盖优化器超参数:")
                        for key, value in self.override_hypers.items():
                            print(f"   {key}: {value}")
                    
                    # 更新所有参数组的超参数
                    for param_group in self.learn.opt.param_groups:
                        for key, value in self.override_hypers.items():
                            param_group[key] = value
                
                # 输出恢复的优化器参数以便验证
                self._print_optimizer_params()
                self.loaded = True
            except Exception as e:
                print(f"⚠️  警告: 无法加载优化器状态: {e}")
                print("   将使用新的优化器状态继续训练")
                import traceback
                traceback.print_exc()
                self.loaded = True  # 标记为已尝试，避免重复尝试
    
    def _move_optimizer_state_to_device(self, state_dict, device):
        """将优化器状态字典中的所有张量移动到指定设备"""
        import torch
        
        def move_to_device(obj):
            """递归地将张量移动到目标设备"""
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {key: move_to_device(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(item) for item in obj)
            else:
                return obj
        
        return move_to_device(state_dict)
    
    def _print_optimizer_params(self):
        """打印优化器参数以便验证"""
        if is_main_process():
            print("✅ 优化器状态已恢复（训练开始前）")
            try:
                # 显示参数组数量
                print(f"📊 恢复的优化器参数:")
                print(f"   - 参数组数量: {len(self.learn.opt.param_groups)}")
                
                # 显示每个参数组的详细信息
                for i, pg in enumerate(self.learn.opt.param_groups):
                    print(f"\n   组 {i}:")
                    print(f"     - lr: {pg.get('lr', 'N/A')}")
                    
                    # FastAI 使用 'mom' 和 'wd'，而不是 'momentum' 和 'weight_decay'
                    momentum = pg.get('mom', pg.get('momentum', 'N/A'))
                    weight_decay = pg.get('wd', pg.get('weight_decay', 'N/A'))
                    
                    print(f"     - mom: {momentum}")
                    print(f"     - wd: {weight_decay}")
                    print(f"     - 参数数量: {len(pg['params'])}")
                
                # 检查是否所有组的 wd 一致
                wd_values = []
                for pg in self.learn.opt.param_groups:
                    wd = pg.get('wd', pg.get('weight_decay', None))
                    if wd is not None:
                        wd_values.append(wd)
                
                if len(set(wd_values)) > 1:
                    print(f"\n   ⚠️  注意: 不同参数组使用不同的权重衰减值")
                    print(f"       值: {set(wd_values)}")
                
            except Exception as e:
                print(f"   无法显示优化器参数: {e}")


class ResumeEpochCallback(Callback):
    """修正从checkpoint恢复时的epoch显示"""
    def __init__(self, resume_from_epoch=0):
        self.resume_from_epoch = resume_from_epoch
    
    def after_epoch(self):
        """在显示metrics时修正epoch数值"""
        if self.resume_from_epoch > 0 and is_main_process():
            actual_epoch = self.learn.epoch + self.resume_from_epoch
            # 在终端输出实际的 epoch 信息
            print(f"  (实际完成的 epoch: {actual_epoch})")


class EarlyStoppingWithEvalCallback(TrackerCallback):
    """早停回调，触发时进行模型评估"""
    def __init__(self, monitor='valid_loss', patience=5, resume_best_metric=None):
        super().__init__(monitor=monitor)
        self.patience = patience
        self.wait = 0
        self.triggered = False  # 标记是否触发了早停
        self.resume_best_metric = resume_best_metric  # 恢复的最佳指标
    
    def before_fit(self):
        """在训练开始前，如果有恢复的 best_metric，使用它"""
        super().before_fit()
        if self.resume_best_metric is not None:
            self.best = self.resume_best_metric
            if is_main_process():
                print(f"✅ 早停回调恢复最佳指标: {self.best:.4f}")
        
    def after_epoch(self):
        super().after_epoch()  # 父类会设置 self.best 和 self.new_best
        
        # 使用父类的 new_best 属性判断是否有改善
        if self.new_best:
            # 有改善，重置等待计数
            self.wait = 0
            if is_main_process():
                print(f"📉 {self.monitor} 改善: {self.best:.4f} (重置早停计数)")
        else:
            # 无改善，增加等待计数
            self.wait += 1

            if self.wait >= self.patience:
                if is_main_process():
                    print(f'\n⚠️  早停触发: {self.monitor} 在 {self.patience} 个 epoch 内没有改善, 最佳 {self.monitor}: {self.best:.4f}')
                self.triggered = True
                raise CancelFitException()


class SaveModelWithEpochCallback(TrackerCallback):
    """保存模型时同时保存当前epoch信息、img_size和arch，并在每个epoch后保存last.pth """
    
    def __init__(self, monitor='valid_loss', fname='best', last_fname='last', with_opt=True, 
                 resume_from_epoch=0, img_size=None, arch=None, resume_best_metric=None, 
                 save_dir=None, save_last=True, upload_to_mlflow=False):
        # 调用父类初始化，处理 monitor 和 comp（valid_loss 越小越好）
        super().__init__(monitor=monitor, comp=np.less)
        
        # 自定义参数
        self.fname = fname
        self.last_fname = last_fname
        self.with_opt = with_opt
        self.resume_from_epoch = resume_from_epoch
        self.img_size = img_size
        self.arch = arch
        self.save_dir = save_dir
        self.save_last = save_last
        self.upload_to_mlflow = upload_to_mlflow
        
        # 如果有恢复的最佳指标，设置初始值（TrackerCallback 使用 self.best）
        if resume_best_metric is not None:
            self.best = resume_best_metric
    
    def _save_model(self, fname, actual_epoch, current_metric=None):
        """统一的模型保存逻辑"""
        # 确定保存路径
        if self.save_dir is not None:
            model_path = Path(self.save_dir) / f'{fname}.pth'
        else:
            model_path = self.learn.path / self.learn.model_dir / f'{fname}.pth'
        
        # 确保目录存在
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取模型状态（处理DDP包装）
        if hasattr(self.learn.model, 'module'):
            model_state = self.learn.model.module.state_dict()
        else:
            model_state = self.learn.model.state_dict()
        
        # 构建保存的状态字典
        state = {
            'model': model_state,
            'epoch': actual_epoch,
            'img_size': self.img_size,
            'arch': self.arch,
            "loss": current_metric,
        }
        
        if self.with_opt:
            # 检查优化器是否存在
            if not hasattr(self.learn, 'opt') or self.learn.opt is None:
                if is_main_process():
                    print("⚠️  优化器还未初始化，跳过保存优化器状态")
            else:
                state['opt'] = self.learn.opt.state_dict()
                
                # # 输出保存的优化器参数
                # if is_main_process():
                #     self._print_save_optimizer_params(state['opt'])
        
        # 保存
        torch.save(state, model_path)
        return model_path, current_metric
    
    def _print_save_optimizer_params(self, opt_state):
        """打印要保存的优化器参数（支持 PyTorch 和 FastAI 格式）"""
        try:
            # 检查优化器状态是否为空或无效
            if not opt_state:
                print("💾 保存优化器状态: ⚠️  优化器状态为空（可能训练还未开始）")
                return
            
            if not isinstance(opt_state, dict):
                print(f"💾 保存优化器状态: ⚠️  优化器状态类型异常 ({type(opt_state)})")
                return
            
            print("💾 保存优化器状态:")
            
            # FastAI 优化器格式：使用 'hypers' 而不是 'param_groups'
            if 'hypers' in opt_state:
                print("   - 格式: FastAI (hypers)")
                hypers = opt_state['hypers']
                
                # hypers 可能是 fastcore.foundation.L 类型或普通列表
                # 都支持 len() 和迭代
                try:
                    hypers_len = len(hypers)
                    print(f"   - 参数组数量: {hypers_len}")
                    
                    for i, hyper_group in enumerate(hypers):
                        print(f"\n   组 {i}:")
                        
                        # FastAI hypers 是字典的字典：{param_name: {hyper_name: value}}
                        if isinstance(hyper_group, dict):
                            print(f"     - 参数数量: {len(hyper_group)}")
                            
                            # 打印前几个参数的详细信息用于调试
                            print(f"     - 参数示例:")
                            for idx, (param_idx, param_hypers) in enumerate(list(hyper_group.items())):
                                print(f"       [{param_idx}]: {param_hypers}")
                            
                        else:
                            print(f"     ⚠️  不是字典类型")
                except Exception as e:
                    print(f"   ⚠️  解析 hypers 失败: {e}")
                    print(f"   hypers 类型: {type(hypers)}")
                
                # 显示 state 信息（如果存在）
                if 'state' in opt_state:
                    state = opt_state['state']
                    if isinstance(state, dict):
                        state_size = len(state)
                    elif hasattr(state, '__len__'):
                        state_size = len(state)
                    else:
                        state_size = f"未知类型 ({type(state).__name__})"
                    print(f"\n   - state 大小: {state_size}")
                
                return
            
            # PyTorch 标准优化器格式：使用 'param_groups'
            if 'param_groups' not in opt_state:
                print(f"   ⚠️  未知的优化器格式")
                print(f"   实际的键: {list(opt_state.keys())}")
                return
            
            print("   - 格式: PyTorch (param_groups)")
            print(f"   - 参数组数量: {len(opt_state['param_groups'])}")
            
            # 显示每个参数组的详细信息
            for i, pg in enumerate(opt_state['param_groups']):
                print(f"\n   组 {i}:")
                
                # 显示关键参数
                lr = pg.get('lr', 'N/A')
                wd = pg.get('wd', 'N/A')
                weight_decay = pg.get('weight_decay', 'N/A')
                mom = pg.get('mom', pg.get('momentum', 'N/A'))
                
                print(f"     - lr: {lr}")
                print(f"     - mom: {mom}")
                print(f"     - wd: {wd}")
                print(f"     - weight_decay: {weight_decay}")
                print(f"     - 参数数量:                 {len(pg.get('params', []))}")
                
                # 显示所有键（用于调试）
                if i == 0:  # 只显示第一个组的所有键
                    other_keys = [k for k in pg.keys() if k not in ['lr', 'wd', 'weight_decay', 'mom', 'momentum', 'params']]
                    if other_keys:
                        print(f"     - 其他键: {', '.join(other_keys)}")
            
            # 警告：wd 和 weight_decay 不一致
            for i, pg in enumerate(opt_state['param_groups']):
                wd = pg.get('wd', None)
                weight_decay = pg.get('weight_decay', None)
                
                if wd is not None and weight_decay is not None:
                    if abs(float(wd) - float(weight_decay)) > 1e-6:
                        print(f"\n   ⚠️  组 {i}: wd ({wd}) 和 weight_decay ({weight_decay}) 不一致")
            
        except Exception as e:
            print(f"   无法显示优化器参数: {e}")
    
    def _upload_to_mlflow(self, model_path):
        """上传模型到 MLflow（带签名和示例）"""
        try:
            from mlflow.models.signature import infer_signature

            # 获取原始模型（去除 DDP 包装）
            if hasattr(self.learn.model, 'module'):
                model_to_log = self.learn.model.module
            else:
                model_to_log = self.learn.model
            
            # 创建随机示例输入（最安全的方式，避免任何DataLoader副作用）
            # 形状: [batch_size, channels, height, width]
            # 需要与模型在同一设备上
            device = next(model_to_log.parameters()).device
            input_example = torch.randn(1, 3, self.img_size, self.img_size, device=device)

            # temp_dl. 
            # 运行一次前向传播获取输出
            with torch.no_grad():
                output = model_to_log(input_example)
            
            # 推断模型签名
            signature = infer_signature(
                input_example.cpu().numpy(),
                output.cpu().numpy()
            )
            
            # 使用 mlflow.pytorch.log_model 上传 PyTorch 模型（带签名和示例）
            mlflow.pytorch.log_model(
                pytorch_model=model_to_log,
                name="best",
                signature=signature,
                input_example=input_example.cpu().numpy(),
                registered_model_name=None,  # 不自动注册到 Model Registry
            )
            print(f"   📤 已上传 PyTorch 模型到 MLflow（包含模型签名）")
            
        except Exception as sig_error:
            # 如果签名推断失败，回退到不带签名的上传
            print(f"   ⚠️  模型签名推断失败: {sig_error}，使用不带签名的方式上传")
            mlflow.pytorch.log_model(
                pytorch_model=model_to_log,
                name="best",
                registered_model_name=None
            )
            print(f"   📤 已上传 PyTorch 模型到 MLflow")

    def _get_monitor_value(self):

        # FastAI的metric_names第一个是'epoch'，但values中不包含epoch
        # 所以需要从metric_names[1:]开始匹配
        last_values = self.learn.recorder.values[-1]

        metric_names = self.learn.recorder.metric_names[1:]  # 跳过 'epoch'
        
        try:
            metric_idx = metric_names.index(self.monitor)
            return last_values[metric_idx]
        except (ValueError, IndexError) as e:
            # 如果找不到指标或索引超出范围，打印调试信息并跳过
            print(f"⚠️  无法找到监控指标 '{self.monitor}'")
            print(f"   可用指标: {metric_names}")
            print(f"   values长度: {len(last_values)}")
            return None
        
    def after_epoch(self):
        """每个 epoch 后保存模型到本地"""
        # 只在主进程执行保存
        if not is_main_process():
            return

        super().after_epoch()
        
        # 注意：显式调用父类方法，避免 FastAI 的属性查找问题
        current_metric = self._get_monitor_value()
        if current_metric is None:
            print(f"⚠️  无法获取监控指标 '{self.monitor}'，跳过保存")
            return

        # 计算实际的epoch（考虑resume）
        actual_epoch = self.learn.epoch + self.resume_from_epoch
        
        # 1. 保存 best 模型到本地（使用 TrackerCallback 的 new_best() 判断）
        if self.new_best:
            model_path, _ = self._save_model(self.fname, actual_epoch, current_metric)
            print(f"✅ 保存最佳模型到本地: {self.fname}.pth, Epoch: {actual_epoch}, {self.monitor}: {current_metric:.4f}")
            
            # 🆕 立即上报所有best指标到MLflow
            if self.upload_to_mlflow:
                try:
                    import mlflow
                    if mlflow.active_run():
                        # 获取当前epoch的所有指标
                        last_values = self.learn.recorder.values[-1]
                        metric_names = self.learn.recorder.metric_names[1:]  # 跳过 'epoch'
                        
                        # 构建best指标字典
                        best_metrics = {'best/epoch': int(actual_epoch)}
                        
                        # 遍历所有指标，添加到best指标中
                        for idx, name in enumerate(metric_names):
                            if idx < len(last_values):
                                value = last_values[idx]
                                # 使用best/前缀命名
                                best_metrics[f'best/{name}'] = float(value)
                        
                        # 上报到MLflow
                        mlflow.log_metrics(best_metrics)
                        
                        # # 打印上报的指标
                        # print(f"   📊 已上报best指标到MLflow:")
                        # for key, value in best_metrics.items():
                        #     if key != 'best/epoch':
                        #         print(f"      - {key}: {value:.4f}")
                        #     else:
                        #         print(f"      - {key}: {value}")
                except Exception as e:
                    print(f"   ⚠️  上报best指标到MLflow失败: {e}")
        
        # 2. 始终保存 last 模型（如果启用）
        if self.save_last:
            self._save_model(self.last_fname, actual_epoch, current_metric)
    
    def after_fit(self):
        """训练结束后上传最终的 best 模型到 MLflow"""
        if not is_main_process() or not self.upload_to_mlflow:
            return
        
        # 确定 best 模型路径
        if self.save_dir is not None:
            best_path = Path(self.save_dir) / f'{self.fname}.pth'
        else:
            best_path = self.learn.path / self.learn.model_dir / f'{self.fname}.pth'
        
        # 上传最终的 best 模型
        if best_path.exists():
            print(f"\n📤 上传最终 best 模型到 MLflow...")
            try:
                self._upload_to_mlflow(best_path)
                print(f"✅ 成功上传 best 模型到 MLflow")
            except Exception as e:
                print(f"⚠️  上传到 MLflow 失败: {e}")
        else:
            print(f"⚠️  best 模型不存在，跳过上传: {best_path}")



class DistributedValidationDiagnosticCallback(Callback):
    """
    诊断多GPU分布式训练中的验证loss计算
    
    用途:
        在多GPU训练时，在batch级别收集每个GPU的原始loss
        帮助发现数据分布不均或loss聚合问题
    
    原理:
        FastAI/Accelerate在验证结束时会自动gather和broadcast loss，
        所以在after_epoch()时所有GPU已经看到相同的聚合后的loss。
        
        本callback在after_batch()时收集每个GPU的原始batch loss，
        在after_epoch()时打印诊断信息（此时recorder已更新）。
    
    用法:
        callbacks.append(DistributedValidationDiagnosticCallback(verbose=True))
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.epoch_losses = []  # 记录每个epoch各GPU的平均loss
        self.current_epoch_batch_losses = []  # 当前epoch验证batch的loss列表
        self.diagnostic_data = None  # 保存诊断数据，在after_epoch打印
        
    def before_validate(self):
        """验证开始前，重置当前epoch的batch losses"""
        self.current_epoch_batch_losses = []
        self.diagnostic_data = None
    
    def after_batch(self):
        """在每个验证batch后收集loss（尝试在Accelerate聚合之前）"""
        if self.training:  # 只在验证阶段收集
            return
        
        try:
            # 获取当前batch的loss
            if hasattr(self.learn, 'loss') and self.learn.loss is not None:
                batch_loss = float(self.learn.loss.detach().cpu())
                self.current_epoch_batch_losses.append(batch_loss)
        except Exception as e:
            pass  # 静默失败，不影响训练
        
    def after_validate(self):
        """验证结束后，收集所有GPU的batch losses（但不打印，等after_epoch）"""
        try:
            import torch.distributed as dist
            if not dist.is_available() or not dist.is_initialized():
                return
            
            if len(self.current_epoch_batch_losses) == 0:
                return
            
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # 计算当前GPU在所有验证batch上的平均loss
            local_avg_loss = sum(self.current_epoch_batch_losses) / len(self.current_epoch_batch_losses)
            local_num_batches = len(self.current_epoch_batch_losses)
            
            # 收集所有GPU的平均loss和batch数
            all_avg_losses = [torch.zeros(1).cuda() for _ in range(world_size)]
            all_num_batches = [torch.zeros(1).cuda() for _ in range(world_size)]
            
            dist.all_gather(all_avg_losses, torch.tensor([local_avg_loss]).cuda())
            dist.all_gather(all_num_batches, torch.tensor([float(local_num_batches)]).cuda())
            
            # 保存数据，稍后在after_epoch打印
            if rank == 0:
                avg_losses = [t.item() for t in all_avg_losses]
                num_batches = [int(t.item()) for t in all_num_batches]
                
                self.epoch_losses.append(avg_losses)
                
                # 计算加权平均（考虑每个GPU处理的batch数可能不同）
                total_batches = sum(num_batches)
                weighted_avg = sum(l * n for l, n in zip(avg_losses, num_batches)) / total_batches if total_batches > 0 else 0
                simple_avg = sum(avg_losses) / len(avg_losses) if len(avg_losses) > 0 else 0
                
                # 保存诊断数据
                self.diagnostic_data = {
                    'avg_losses': avg_losses,
                    'num_batches': num_batches,
                    'weighted_avg': weighted_avg,
                    'simple_avg': simple_avg,
                    'std_loss': np.std(avg_losses),
                    'min_loss': min(avg_losses),
                    'max_loss': max(avg_losses),
                    'total_batches': total_batches
                }
                
        except Exception as e:
            if is_main_process() and self.verbose:
                print(f"⚠️  分布式验证诊断数据收集失败: {e}")
                import traceback
                traceback.print_exc()
    
    def after_epoch(self):
        """在epoch结束后打印诊断信息（此时recorder已更新为当前epoch）"""
        if not is_main_process() or not self.verbose or self.diagnostic_data is None:
            return
        
        try:
            data = self.diagnostic_data
            avg_losses = data['avg_losses']
            num_batches = data['num_batches']
            weighted_avg = data['weighted_avg']
            simple_avg = data['simple_avg']
            std_loss = data['std_loss']
            min_loss = data['min_loss']
            max_loss = data['max_loss']
            total_batches = data['total_batches']
            
            print(f"\n{'='*80}")
            print(f"🔍 Epoch {self.learn.epoch} 分布式验证诊断 (Batch级别):")
            print(f"{'='*80}")
            
            # 打印每个GPU的信息
            print("各GPU的验证信息:")
            for i, (loss, n_batch) in enumerate(zip(avg_losses, num_batches)):
                marker = ""
                if loss == min_loss:
                    marker = " ← 最低"
                elif loss == max_loss:
                    marker = " ← 最高"
                print(f"  GPU{i}: avg_loss={loss:.6f}, batches={n_batch}{marker}")
            
            print(f"\n统计信息:")
            print(f"  简单平均: {simple_avg:.6f}")
            print(f"  加权平均: {weighted_avg:.6f} (考虑batch数)")
            print(f"  标准差: {std_loss:.6f}")
            print(f"  最小值: {min_loss:.6f}")
            print(f"  最大值: {max_loss:.6f}")
            print(f"  极差: {max_loss - min_loss:.6f}")
            print(f"  总batch数: {total_batches}")
            
            # 警告检查
            if std_loss > simple_avg * 0.2:
                print(f"\n⚠️  警告: GPU间loss差异较大 (std={std_loss:.6f} > 20%*avg)")
                print(f"   可能原因:")
                print(f"     - 验证集数据分布不均（不同GPU处理不同类别）")
                print(f"     - 某些类别特别难，导致对应GPU的loss高")
                print(f"   建议:")
                print(f"     - 检查验证集是否已打乱 (应该看到 '🔀 打乱验证集...')")
                print(f"     - 检查各GPU处理的batch数是否均衡")
            elif std_loss > simple_avg * 0.1:
                print(f"\n⚠️  提示: GPU间loss有一定差异 (std={std_loss:.6f})")
                print(f"   这在多GPU训练中是正常的，但建议监控")
            else:
                print(f"\n✅ GPU间loss分布均匀 (std={std_loss:.6f} < 10%*avg)")
                print(f"   验证集打乱修复已生效！")
            
            # 检查batch数分布
            if max(num_batches) - min(num_batches) > 1:
                print(f"\n⚠️  提示: GPU间batch数不完全均衡 (差{max(num_batches) - min(num_batches)}个)")
                print(f"   这是正常的（数据总数不能被GPU数整除）")
            
            # 对比FastAI/Accelerate报告的值（现在应该是当前epoch的值）
            if hasattr(self.learn.recorder, 'values') and len(self.learn.recorder.values) > 0:
                last_metrics = self.learn.recorder.values[-1]
                if len(last_metrics) >= 2:
                    reported_loss = float(last_metrics[1])
                    print(f"\n对比FastAI报告 (当前epoch):")
                    print(f"  FastAI报告的valid_loss: {reported_loss:.6f}")
                    print(f"  我们计算的加权平均: {weighted_avg:.6f}")
                    diff = abs(reported_loss - weighted_avg)
                    if diff > 0.01:
                        print(f"  ⚠️  差异: {diff:.6f}")
                        print(f"     说明: FastAI/Accelerate使用了不同的聚合方式")
                    else:
                        print(f"  ✅ 基本一致 (差异 < 0.01)")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            if is_main_process() and self.verbose:
                print(f"⚠️  分布式验证诊断打印失败: {e}")
                import traceback
                traceback.print_exc()
    
    def after_fit(self):
        """训练结束后打印总结"""
        if not is_main_process() or not self.verbose or len(self.epoch_losses) == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 分布式验证诊断总结")
        print(f"{'='*80}")
        
        # 计算整体统计
        all_epochs_losses = np.array(self.epoch_losses)  # shape: (n_epochs, n_gpus)
        n_epochs, n_gpus = all_epochs_losses.shape
        
        print(f"总共训练: {n_epochs} epochs, {n_gpus} GPUs")
        
        # 每个GPU的平均loss
        print(f"\n各GPU的平均valid_loss (所有epochs):")
        for gpu_id in range(n_gpus):
            gpu_avg = all_epochs_losses[:, gpu_id].mean()
            gpu_std = all_epochs_losses[:, gpu_id].std()
            print(f"  GPU{gpu_id}: {gpu_avg:.6f} ± {gpu_std:.6f}")
        
        # 每个epoch的GPU间差异
        epoch_stds = [np.std(epoch_losses) for epoch_losses in all_epochs_losses]
        avg_std = np.mean(epoch_stds)
        max_std = max(epoch_stds)
        
        print(f"\nGPU间loss差异:")
        print(f"  平均标准差: {avg_std:.6f}")
        print(f"  最大标准差: {max_std:.6f}")
        
        if avg_std > 0.1:
            print(f"\n⚠️  警告: 整体GPU间差异较大")
            print(f"   建议:")
            print(f"     - 确认验证集已打乱")
            print(f"     - 检查数据加载是否正常")
        else:
            print(f"\n✅ GPU间loss分布整体均匀")
            print(f"   验证集打乱修复有效！")
        
        # 趋势分析
        if n_epochs > 3:
            early_avg_std = np.mean(epoch_stds[:3])
            late_avg_std = np.mean(epoch_stds[-3:])
            print(f"\n趋势分析:")
            print(f"  前3个epoch平均标准差: {early_avg_std:.6f}")
            print(f"  后3个epoch平均标准差: {late_avg_std:.6f}")
            if late_avg_std < early_avg_std * 0.8:
                print(f"  ✅ GPU间差异随训练减小（模型学习均衡）")
            elif late_avg_std > early_avg_std * 1.2:
                print(f"  ⚠️  GPU间差异随训练增大（可能过拟合不均）")
            else:
                print(f"  → GPU间差异保持稳定")
        
        print(f"{'='*80}\n")
