"""
Implementation of 4D PyTorch compatible layers.
"""

import torch
from math import ceil
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union

from torch4d.functional import max_pool4d, drop_block4d, spatial_dropout4d

class MaxPool4d(nn.Module):
    """
    See :func:`max_pool4d`.
    """

    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple]) -> None:
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature maps on which the maxpool operation will be applied (input assumed to be of shape: Tensor[N, C, K, D, H, W])
        
        Returns:
            returned_tensor (Tensor): The tensor after max pool operation is executed.
        """
        returned_tensor, indices = max_pool4d(input, self.kernel_size, self.stride)
        return returned_tensor

class DropBlock4d(nn.Module):
    """
    See :func:`drop_block4d`.
    """

    def __init__(self, p: float, block_size: Union[int, tuple], inplace: bool = False, eps: float = 1e-06) -> None:
        super().__init__()

        self.p = p
        self.block_size = block_size
        self.inplace = inplace
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly dropped.

        Returns:
            _ (Tensor): The tensor after Dropblock layer. 
        """
        return drop_block4d(input, self.p, self.block_size, self.inplace, self.eps, self.training) # self.training from super class initialization

class SpatialDropout4d(nn.Module):
    """
    Spatial dropout for 4D tensors, uses inverted dropout implementation (see: https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch)
    """
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """
        SpatialDropout4D initializer
        Args:
            p: probability to drop (likelihood of channel to be dropped)
            inplace: if True, do operation in place.
        Returns:
            None
        """
        super(SpatialDropout4d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Spatial dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.inplace = inplace
        
    def forward(self, X: Tensor) -> Tensor:
        """
        See :func:`spatial_dropout4d`.
        """
        return spatial_dropout4d(X, self.p, self.inplace, self.training) # self.training from super class intialization 