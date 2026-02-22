'''
From https://github.com/wjq-learning/CBraMod
Original code released under the MIT License
'''

import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def to_tensor(array):
    # Use torch.tensor to force a copy and avoid sharing the underlying numpy buffer.
    # This prevents 'deallocated bytearray object has exported buffers' when arrays come
    # from transient LMDB transaction buffers or other temporary views.
    return torch.tensor(array, dtype=torch.float32)


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)