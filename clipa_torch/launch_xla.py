"""
Adapatation of (pre-elastic) torch.distributed.launch for pytorch xla.

`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.

Based on https://github.com/huggingface/pytorch-image-models/blob/bits_and_tpu/launch_xla.py
"""


import sys
import subprocess
import importlib
import os
from argparse import ArgumentParser, REMAINDER
from typing import Optional, IO

import torch_xla.distributed.xla_multiprocessing as xmp


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch helper utility"
                    "that will spawn up multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--num-devices", type=int, default=1,
                        help="The number of XLA devices to use for distributed training")

    # positional
    parser.add_argument(
        "module", type=str,
        help="The full path to the single device training script to be launched"
             "in parallel, followed by all the arguments for the training script")

    # rest from the training program
    parser.add_argument('script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # set PyTorch distributed related environmental variables
    # current_env = os.environ.copy()
    # current_env["MASTER_ADDR"] = args.master_addr
    # current_env["MASTER_PORT"] = str(args.master_port)
    # current_env["WORLD_SIZE"] = str(dist_world_size)
    # if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
    #    current_env["OMP_NUM_THREADS"] = str(1)

    mod = importlib.import_module(args.module)

    sys.argv = [args.module] + args.script_args

    xmp.spawn(mod._mp_entry, args=(), nprocs=args.num_devices)


if __name__ == "__main__":
    main()