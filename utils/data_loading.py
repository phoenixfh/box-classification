"""
数据加载工具函数
"""

import os
import torch.distributed as dist


def is_main_process():
    """
    判断是否为主进程（分布式训练支持）
    
    Returns:
        bool: 如果是主进程或单进程训练返回 True，否则返回 False
    """
    # 优先检查环境变量（torchrun 会设置这些）
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    
    # 其次检查 torch.distributed
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    
    # 默认单进程模式
    return True
