import pytest
import torch
import torch.nn as nn
from thop import profile


class TestUtils:
    def test_matmul_case2(self):
        n, in_c, out_c = 1, 100, 200
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
        print(flops, params)
        assert flops == n * in_c * out_c

    def test_matmul_case2(self):
        for i in range(10):
            n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
            net = nn.Linear(in_c, out_c)
            flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
            print(flops, params)
            assert flops == n * in_c * out_c
    
    def test_conv2d(self):
        n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
        print(flops, params)
        assert flops == n * in_c * out_c
    
