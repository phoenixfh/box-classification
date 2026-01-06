"""
Logging utilities for distributed training
"""
import os
import torch.distributed as dist


def is_main_process():
    """Check if current process is the main process"""
    # 优先使用环境变量（最可靠）
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    # 如果没有环境变量，检查distributed状态
    if dist.is_initialized():
        return dist.get_rank() == 0
    # 默认情况（非分布式）
    return True


def print_main(*args, **kwargs):
    """Print only from main process in distributed training"""
    if is_main_process():
        print(*args, **kwargs, flush=True)  # 立即刷新输出缓冲区
