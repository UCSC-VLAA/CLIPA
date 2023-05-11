import logging
import torch
from typing import Dict, Any

try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    _HAS_XLA = True
except ImportError as e:
    xm = None
    torch_xla = None
    _HAS_XLA = False

USE_XLA = False


def is_xla_available(xla_device_type=None):
    if not _HAS_XLA:
        return False
    supported_devs = xm.get_xla_supported_devices(devkind=xla_device_type)
    return len(supported_devs) >= 1


def initialize_device(args) -> torch.device:
    if is_xla_available():
        # XLA supports more than just TPU, will search in order TPU, GPU, CPU
        device = xm.xla_device()
        global USE_XLA
        USE_XLA = True
        assert not args.horovod, 'the pytorch xla implementation does not support horovod!'
    elif torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
        device = torch.device(device)
    else:
        device = 'cpu'
        device = torch.device(device)

    args.device = device

    return device

def use_xla():
    return USE_XLA


#######################
# acknowledgement: https://github.com/huggingface/pytorch-image-models
def state_dict_apply(state_dict: Dict[str, Any], apply_fn, select_fn=lambda x: isinstance(x, torch.Tensor)):
    out_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            out_dict[k] = state_dict_apply(v, apply_fn, select_fn)
        else:
            out_dict[k] = apply_fn(v) if select_fn(v) else v
    return out_dict


def state_dict_to_cpu(state: Dict[str, Any]):
    cpu_state = state_dict_apply(state, apply_fn=lambda x: x.cpu())
    return cpu_state
#######################