import argparse

import torch
import torch.nn as nn

multiply_adds = 1


def count_conv2d(m, x, y):
	# TODO: add support for pad and dilation
	x = x[0]

	cin = m.in_channels
	cout = m.out_channels
	kh, kw = m.kernel_size
	batch_size = x.size()[0]

	out_w = y.size(2) // m.stride[0]
	out_h = y.size(3) // m.stride[1]

	# ops per output element
	# kernel_mul = kh * kw * cin
	# kernel_add = kh * kw * cin - 1
	kernel_ops = multiply_adds * kh * kw * cin // m.groups
	bias_ops = 1 if m.bias is not None else 0
	ops_per_element = kernel_ops + bias_ops

	# total ops
	# num_out_elements = y.numel()
	output_elements = batch_size * out_w * out_h * cout
	total_ops = output_elements * ops_per_element

	# in case same conv is used multiple times
	m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
	x = x[0]

	nelements = x.numel()
	total_sub = nelements
	total_div = nelements
	total_ops = total_sub + total_div

	m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
	x = x[0]

	nelements = x.numel()
	total_ops = nelements

	m.total_ops += torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
	x = x[0]

	batch_size, nfeatures = x.size()

	total_exp = nfeatures
	total_add = nfeatures - 1
	total_div = nfeatures
	total_ops = batch_size * (total_exp + total_add + total_div)

	m.total_ops += torch.Tensor([int(total_ops)])


def count_maxpool(m, x, y):
	kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
	num_elements = y.numel()
	total_ops = kernel_ops * num_elements

	m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
	total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
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
