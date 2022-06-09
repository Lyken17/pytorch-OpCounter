import pytest
import torch
import torch.nn as nn
from thop import profile


class TestUtils:
    def test_relu(self):
        n, in_c, out_c = 1, 100, 200
        data = torch.randn(n, in_c)
        net = nn.ReLU()
        flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
        print(flops, params)
        assert flops == 0

    
