"""
超参数调优策略实现

支持网格搜索、随机搜索和贝叶斯优化（Optuna）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator, Optional
import itertools
import random
import numpy as np


class SearchStrategy(ABC):
    """搜索策略基类"""
    
    def __init__(self, search_space: Dict[str, Any], n_trials: Optional[int] = None):
        """
        Args:
            search_space: 参数搜索空间定义
            n_trials: 试验次数（仅用于随机搜索和Optuna）
        """
        self.search_space = search_space
        self.n_trials = n_trials
    
    @abstractmethod
    def generate_trials(self) -> Iterator[Dict[str, Any]]:
        """生成参数组合迭代器"""
        pass
    
    @abstractmethod
    def get_total_trials(self) -> int:
        """获取总试验次数"""
        pass


class GridSearchStrategy(SearchStrategy):
    """网格搜索：遍历所有参数组合"""
    
    def generate_trials(self) -> Iterator[Dict[str, Any]]:
        """生成所有参数组合（笛卡尔积）"""
        # 解析搜索空间
        param_names = []
        param_values = []
        
        for param_name, param_config in self.search_space.items():
            param_names.append(param_name)
            values = self._parse_param_config(param_config)
            param_values.append(values)
        
        # 生成笛卡尔积
        for combination in itertools.product(*param_values):
            trial = dict(zip(param_names, combination))
            yield trial
    
    def get_total_trials(self) -> int:
        """计算总试验次数"""
        total = 1
        for param_config in self.search_space.values():
            values = self._parse_param_config(param_config)
            total *= len(values)
        return total
    
    def _parse_param_config(self, config: Any) -> List[Any]:
        """解析参数配置，返回候选值列表"""
        if isinstance(config, list):
            # 离散值列表: [0.001, 0.01, 0.1]
            return config
        elif isinstance(config, dict):
            param_type = config.get('type', 'choice')
            
            if param_type == 'choice':
                # 分类选择: {type: choice, values: [...]}
                return config['values']
            elif param_type in ['int', 'uniform', 'log_uniform']:
                # 整数范围: {min: 32, max: 256, step: 32}
                min_val = float(config['min'])
                max_val = float(config['max'])
                step = config.get('step', 1)
                
                if param_type == 'int':
                    return list(range(int(min_val), int(max_val) + 1, int(step)))
                else:
                    # 对于连续值，生成等间隔的采样点
                    n_points = config.get('n_points', 10)
                    if param_type == 'log_uniform':
                        # 对数空间采样
                        return list(np.logspace(np.log10(min_val), np.log10(max_val), n_points))
                    else:
                        # 线性空间采样
                        return list(np.linspace(min_val, max_val, n_points))
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        else:
            # 单个值
            return [config]


class RandomSearchStrategy(SearchStrategy):
    """随机搜索：从参数空间中随机采样"""
    
    def __init__(self, search_space: Dict[str, Any], n_trials: int, seed: int = 42):
        super().__init__(search_space, n_trials)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_trials(self) -> Iterator[Dict[str, Any]]:
        """随机采样参数组合"""
        for _ in range(self.n_trials):
            trial = {}
            for param_name, param_config in self.search_space.items():
                trial[param_name] = self._sample_param(param_config)
            yield trial
    
    def get_total_trials(self) -> int:
        return self.n_trials
    
    def _sample_param(self, config: Any) -> Any:
        """从参数配置中采样单个值"""
        if isinstance(config, list):
            # 离散值列表
            return random.choice(config)
        elif isinstance(config, dict):
            param_type = config.get('type', 'choice')
            
            if param_type == 'choice':
                return random.choice(config['values'])
            elif param_type == 'int':
                min_val = int(config['min'])
                max_val = int(config['max'])
                step = int(config.get('step', 1))
                # 在范围内随机选择
                n_steps = (max_val - min_val) // step
                return min_val + random.randint(0, n_steps) * step
            elif param_type == 'uniform':
                return random.uniform(float(config['min']), float(config['max']))
            elif param_type == 'log_uniform':
                log_min = np.log10(float(config['min']))
                log_max = np.log10(float(config['max']))
                return 10 ** random.uniform(log_min, log_max)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        else:
            return config


class OptunaSearchStrategy(SearchStrategy):
    """贝叶斯优化：使用Optuna的TPE算法"""
    
    def __init__(self, search_space: Dict[str, Any], n_trials: int, 
                 metric_name: str = 'valid_loss', mode: str = 'minimize'):
        super().__init__(search_space, n_trials)
        self.metric_name = metric_name
        self.mode = mode
        
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install it with: pip install optuna>=3.0.0"
            )
        
        # 创建Optuna study
        direction = 'minimize' if mode == 'minimize' else 'maximize'
        self.study = optuna.create_study(direction=direction)
        self.trial_results = []
    
    def generate_trials(self) -> Iterator[Dict[str, Any]]:
        """使用Optuna生成参数组合"""
        for trial_idx in range(self.n_trials):
            trial = self.study.ask()
            params = {}
            
            for param_name, param_config in self.search_space.items():
                params[param_name] = self._suggest_param(trial, param_name, param_config)
            
            # 存储trial对象，用于稍后报告结果
            params['_optuna_trial'] = trial
            params['_trial_idx'] = trial_idx
            
            yield params
    
    def report_result(self, trial_params: Dict[str, Any], metric_value: float):
        """向Optuna报告试验结果"""
        trial = trial_params.get('_optuna_trial')
        if trial:
            self.study.tell(trial, metric_value)
    
    def get_total_trials(self) -> int:
        return self.n_trials
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最佳参数"""
        if not self.study.trials:
            return {}
        return self.study.best_params
    
    def _suggest_param(self, trial, param_name: str, config: Any) -> Any:
        """使用Optuna的suggest方法采样参数"""
        if isinstance(config, list):
            # 离散值列表
            return trial.suggest_categorical(param_name, config)
        elif isinstance(config, dict):
            param_type = config.get('type', 'choice')
            
            if param_type == 'choice':
                return trial.suggest_categorical(param_name, config['values'])
            elif param_type == 'int':
                step = config.get('step', 1)
                return trial.suggest_int(param_name, int(config['min']), int(config['max']), step=step)
            elif param_type == 'uniform':
                return trial.suggest_float(param_name, float(config['min']), float(config['max']))
            elif param_type == 'log_uniform':
                return trial.suggest_float(param_name, float(config['min']), float(config['max']), log=True)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        else:
            # 固定值
            return config


def parse_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析和验证搜索空间配置
    
    Args:
        config: 配置字典，包含 search_space 字段
        
    Returns:
        解析后的搜索空间
    """
    if 'search_space' not in config:
        raise ValueError("Configuration must contain 'search_space' field")
    
    search_space = config['search_space']
    
    # 验证每个参数配置
    for param_name, param_config in search_space.items():
        if isinstance(param_config, dict):
            # 检查必需字段
            if 'type' in param_config:
                param_type = param_config['type']
                if param_type in ['int', 'uniform', 'log_uniform']:
                    if 'min' not in param_config or 'max' not in param_config:
                        raise ValueError(
                            f"Parameter '{param_name}' with type '{param_type}' "
                            f"must have 'min' and 'max' fields"
                        )
                elif param_type == 'choice':
                    if 'values' not in param_config:
                        raise ValueError(
                            f"Parameter '{param_name}' with type 'choice' "
                            f"must have 'values' field"
                        )
    
    return search_space


def create_search_strategy(config: Dict[str, Any]) -> SearchStrategy:
    """
    根据配置创建搜索策略
    
    Args:
        config: 配置字典，包含 strategy, search_space, n_trials 等字段
        
    Returns:
        SearchStrategy 实例
    """
    strategy_name = config.get('strategy', 'grid').lower()
    search_space = parse_search_space(config)
    n_trials = config.get('n_trials')
    
    if strategy_name == 'grid':
        return GridSearchStrategy(search_space)
    elif strategy_name == 'random':
        if not n_trials:
            raise ValueError("Random search requires 'n_trials' parameter")
        seed = config.get('seed', 42)
        return RandomSearchStrategy(search_space, n_trials, seed)
    elif strategy_name == 'optuna':
        if not n_trials:
            raise ValueError("Optuna search requires 'n_trials' parameter")
        metric = config.get('metric', 'valid_loss')
        mode = config.get('mode', 'minimize')
        return OptunaSearchStrategy(search_space, n_trials, metric, mode)
    else:
        raise ValueError(f"Unknown search strategy: {strategy_name}")
