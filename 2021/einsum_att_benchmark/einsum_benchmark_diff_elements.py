import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Callable, List
import torch.nn.functional as F
import time
import random
import numpy as np


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    iterations = 100000
    warmup_iter = 30
    dims = [20, 15, 32, 64]
    _a = [torch.rand(dims[2:]) for _ in range(iterations + warmup_iter)]
    _b = [torch.rand(dims[-1], dims[-2]) for _ in range(iterations + warmup_iter)]
    _c = [torch.rand(dims) for _ in range(iterations + warmup_iter)]
    _e = [torch.rand(dims[:2]+[dims[-1]]+[dims[-2]]) for _ in range(iterations + warmup_iter)]
    _f = [torch.rand(dims[:2]+[dims[-2]]+[dims[-1]]) for _ in range(iterations + warmup_iter)]
    _g = [torch.rand([dims[-1]] + dims[1:]) for _ in range(iterations + warmup_iter)]
    _h = [torch.rand([dims[-1]] + dims[1:]) for _ in range(iterations + warmup_iter)]
    device = ['cpu', 'cuda']

    def benchmark_2(a, b, f, device):
        for i in range(iterations + warmup_iter):
            if i == warmup_iter:
                t0 = time.time()
            f(a[i].to(device), b[i].to(device))
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        return t1-t0

    def benchmark_1(a, f, device):
        for i in range(iterations + warmup_iter):
            if i == warmup_iter:
                t0 = time.time()
            f(a[i].to(device))
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        return t1-t0

    for d in device:
        if not torch.cuda.is_available() and d == 'cuda':
            print('Skipping cuda: cuda is not available')
            break
        print('Using device {}'.format(d))

        # 2-dim matmul
        f = lambda x, y: torch.matmul(x, y)
        print('{} (2 dim) matmul took'.format(iterations), benchmark_2(_a, _b, f, d))

        # 2-dim einsum
        f = lambda x, y: torch.einsum('ik,kj->ij', x, y)
        print('{} (2 dim) einsum matmul took'.format(iterations), benchmark_2(_a, _b, f, d))

        # 4-dim matmul
        f = lambda x, y: torch.matmul(x, y)
        print('{} (4 dim) matmul took'.format(iterations), benchmark_2(_c, _e, f, d))

        # 4-dim einsum
        f = lambda x, y: torch.einsum('abik,abkj->abij', x, y)
        print('{} (4 dim) einsum matmul took'.format(iterations), benchmark_2(_c, _e, f, d))

        # 2-dim permutation
        f = lambda x: x.permute(1, 0)
        print('{} (2 dim) permute took'.format(iterations), benchmark_1(_a, f, d))

        # 2-dim einsum
        f = lambda x: torch.einsum('ik->ki', x)
        print('{} (2 dim) einsum permute took'.format(iterations), benchmark_1(_a, f, d))

        # 4-dim permutation
        f = lambda x : x.permute(3, 0, 2, 1)
        print('{} (4 dim) permute took'.format(iterations), benchmark_1(_c, f, d))

        # 4-dim einsum permutation
        f = lambda x: torch.einsum('abik->kaib', x)
        print('{} (4 dim) einsum permute took'.format(iterations), benchmark_1(_c, f, d))

        # 4-dim matmul and permute
        f = lambda x, y: torch.matmul(x, y).permute(2, 3, 1, 0)
        print('{} (4 dim) matmul and permute took'.format(iterations), benchmark_2(_c, _e, f, d))

        # 4-dim einsum matmul and permute
        f = lambda x, y: torch.einsum('abik, abkj->ijba', x, y)
        print('{} (4 dim) einsum matmul and permute took'.format(iterations), benchmark_2(_c, _e, f, d))
S
