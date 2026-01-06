"""
回调工具函数
"""
import torch.distributed as dist


def is_main_process() -> bool:
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_main(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)
