import torch
from contextlib import suppress

from .device_env_factory import use_xla

def get_autocast(precision):
    assert not (use_xla() and 'amp' in precision),\
        'currently pytorch xla does not support amp training!'
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
