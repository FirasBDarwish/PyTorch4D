""" Testing the torch4d.torch4d Modules """

import pytest
from unittest.mock import patch
from torch4d.torch4d import MaxPool4d, DropBlock4d, SpatialDropout4d
import torch

class Tests:
    """ Class defines tests. """
    def test_MaxPool4d(self):
        pool4d = MaxPool4d(1, 1)
        assert str(type(pool4d)) == "<class 'torch4d.torch4d.MaxPool4d'>"
        assert pool4d.kernel_size == 1
        assert pool4d.stride == 1
        
    def test_DropBlock4d(self):
        dropblock4d = DropBlock4d(0.2, 1)
        assert str(type(dropblock4d)) == "<class 'torch4d.torch4d.DropBlock4d'>"
        assert dropblock4d.p == 0.2
        assert dropblock4d.block_size == 1
        assert dropblock4d.inplace == False
    
    def test_SpatialDropout4d(self):
        spatialdropout4d = SpatialDropout4d(p = 0.2, inplace = False)
        assert str(type(spatialdropout4d)) == "<class 'torch4d.torch4d.SpatialDropout4d'>"
        assert spatialdropout4d.p == 0.2
        assert spatialdropout4d.inplace == False
