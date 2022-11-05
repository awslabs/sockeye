import torch
import argparse
from typing import Optional

def init_device(args: argparse.Namespace, logger=None, local_rank : Optional[int] = None):
    """
    return requested torch device, optionally enabling tf32

    :param args "Device Parameters". args.use_cpu will be set if cuda is not available
    :param logger optional logger.info(msg)
    :param local_rank optional int LOCAL_RANK env for multiple GPU training
    """
    if not torch.cuda.is_available():
        if logger is not None:
            logger.info("CUDA not available, using cpu")
        args.use_cpu = True
    device = torch.device('cpu') if args.use_cpu else torch.device('cuda', args.device_id if local_rank is None else local_rank)
    if not args.use_cpu:
        # Ensure that GPU operations use the correct device by default
        torch.cuda.set_device(device)
        if args.tf32:
            if logger is not None:
                logger.info("CUDA: allow tf32 (float32 but with 10 bits precision)")
                torch.backends.cuda.matmul.allow_tf32 = True
    return device
