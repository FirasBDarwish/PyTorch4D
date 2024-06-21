""" Testing the torch4d.torch4d Modules """

import pytest
from unittest.mock import patch
from torch4d.functional import max_pool4d, drop_block4d
import torch

class Tests:
    """ Class defines tests. """
    def test_max_pool4d(self):
        input = torch.randn(32, 400, 2, 8, 8)
        out, _ = max_pool4d(input, 1, 1)
        assert out.shape == torch.Size([32, 400, 2, 8, 8]) 

    def test_drop_block4d(self):
        input = torch.randn(32, 1, 400, 2, 8, 8)
        inputOldShape = input.shape
        drop_block4d(input, p = 0.2, block_size = (100, 1, 2, 2), inplace = True)
        assert input.shape == inputOldShape