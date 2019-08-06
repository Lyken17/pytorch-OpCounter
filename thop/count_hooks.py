import argparse

import torch
import torch.nn as nn

multiply_adds = 1


def zero_ops(m, x, y):
    m.total_ops += torch.Tensor([int(0)])


def count_convNd(m, x, y):
    x = x[0]

    kernel_ops = m.weight.size()[2:].numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_convNd_ver2(m, x, y):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = (y.size()[:1] + y.size()[2:]).numel()
    # Cout x Cin x Kw x Kh
    kernel_ops = m.weight.nelement()
    if m.bias is not None:
        # Cout x 1
        kernel_ops += + m.bias.nelement()
    # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    m.total_ops += torch.Tensor([int(output_size * kernel_ops)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.Tensor([int(nelements)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
