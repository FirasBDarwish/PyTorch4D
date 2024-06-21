"""
Functional implementations for 4D operations
"""

import torch
from math import ceil
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union

def max_pool4d(input, kernel_size: Union[int, tuple], stride: Union[int, tuple]):
    """
    Implements MaxPool4d (generalization of max_pool3d from PyTorch).

    Args:
        input (Tensor[N, C, K, D, H, W]): The input tensor or 6-dimensions with the first one being its batch i.e. a batch with ``N`` rows.
        kernel_size (tuple): Size of the kernel.
        stride (tuple): Stride of the kernel.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride, stride)
    kk, kd, kh, kw = kernel_size
    dk, dd, dh, dw = stride

    # get all image windows of size (kk, kd, kh, kw) and stride (dk, dd, dh, dw)
    input_windows = input.unfold(2, kk, dk).unfold(3, kd, dd).unfold(4, kh, dh).unfold(5, kw, dw)

    # view the windows as (kk * kd * kh * kw)
    input_windows = input_windows.contiguous().view(*input_windows.size()[:-4], -1)

    max_val, max_idx = input_windows.max(-1)
    
    return max_val, max_idx

def drop_block4d(input: Tensor, p: float, block_size: Union[int, tuple], inplace: bool = False, eps: float = 1e-06, training: bool = True) -> Tensor:
    """
    Implememnts DropBlock4d from '"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/abs/1810.12890>`.

    Args:
        input (Tensor[N, C, K, D, H, W]): The input tensor or 6-dimensions with the first one being its batch i.e. a batch with ``N`` rows.
        p (float): Probability of an element to be dropped. (block or individual pixel?)
        block_size (int, or tuple): Size of the block to drop.
        inplace (bool): If set to ``True``, will do this operation in-place. Default: ``False``.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6.
        training (bool): Apply dropblock if is ``True``. Default: ``True``.

    Returns:
        Tensor[N, C, K, D, H, W]: The randomly zeroed tensor after dropblock.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}.")
    if input.ndim != 6:
        raise ValueError(f"input should be 6-dimensional. Got {input.ndim} dimensions instead.")
    if not training or p == 0.0:
        return input

    if isinstance(block_size, int):
        block_size = [block_size, block_size, block_size, block_size]
    else:
        if len(block_size) != 4:
            raise ValueError("If passing in tuple for block_size, tuple must be of size 4 (for each dimension)")
        block_size = [block_size[0], block_size[1], block_size[2], block_size[3]]

    N, C, K, D, H, W = input.size()
    block_size[0] = min(K, block_size[0])
    block_size[1] = min(D, block_size[1])
    block_size[2] = min(H, block_size[2])
    block_size[3] = min(W, block_size[3])

    # compute the gamma of Bernoulli distribution
    gamma = (p * K * D * H * W) / ((block_size[0]*block_size[1]*block_size[2]*block_size[3]) * ((K - block_size[0] + 1) * (D - block_size[1] + 1) * (H - block_size[2] + 1) * (W - block_size[3] + 1)))

    noise = torch.empty(
        (N, C, K, D, H, W), dtype=input.dtype, device=input.device
    )

    noise.bernoulli_(gamma)

    # pad noise tensor in case that block_size of that dimension is not 0
    if(block_size[0] != 1 or block_size[1] != 1 or block_size[2] != 1 or block_size[3] != 1):
        if block_size[0] != 1:
            noise = F.pad(noise, [0]*6 + [(block_size[0]// 2)]*2, value=0)
        if block_size[1] != 1:
            noise = F.pad(noise, [0]*4 + [(block_size[1]// 2)]*2 + [0]*2, value=0)
        if block_size[2] != 1:
            noise = F.pad(noise, [0]*2 + [(block_size[2]// 2)]*2 + [0]*4, value=0)
        if block_size[3] != 1:
            noise = F.pad(noise, [(block_size[3]// 2)]*2 + [0]*6, value=0)
    noise, _ = max_pool4d(
        noise, stride=(1, 1, 1, 1), kernel_size=(block_size[0], block_size[1], block_size[2], block_size[3])
    )

    # if block_size is even, then remove final element of that tensor (matching dimension as blocksize)
    if block_size[0] % 2 == 0:
        noise = noise[:, :, :-1, :, :, :]
    if block_size[1] % 2 == 0:
        noise = noise[:, :, :, :-1, :, :]
    if block_size[2] % 2 == 0:
        noise = noise[:, :, :, :, :-1, :]
    if block_size[3] % 2 == 0:
        noise = noise[:, :, :, :, :, :-1]

    noise = 1 - noise
    normalize_scale = noise.numel() / (eps + noise.sum())
    if inplace:
        input.mul_(noise).mul_(normalize_scale)
    else:
        input = input * noise * normalize_scale
    return input