import torch
import numpy as np


def counter_parameters(para_list):
    total_params = 0
    for p in para_list:
        total_params += torch.DoubleTensor([p.nelement()])
    return total_params

def counter_zero_ops():
    return torch.DoubleTensor([int(0)])

def counter_conv(bias, kernel_size, output_size):
    """inputs are all numbers!"""
    kernel_ops = 0
    kernel_ops = kernel_size
    if bias is not None:
        kernel_ops += bias
    return torch.DoubleTensor([int(output_size * kernel_ops)])

def counter_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

def counter_relu(input_size: torch.Tensor):
    return torch.DoubleTensor([int(input_size)])

def counter_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])

def counter_avgpool(input_size):
    return torch.DoubleTensor([int(input_size)])

def counter_adap_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])

def counter_upsample(mode: str, output_size):
    total_ops = output_size
    if mode == "linear":
        total_ops *= 5
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "bicubic":
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops *= (ops_solve_A + ops_solve_p)
    elif mode == "trilinear":
        total_ops *= (13 * 2 + 5)
    return torch.DoubleTensor([int(total_ops)])

def counter_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])
def counter_onnx_MatMul(diction,node):
    input1 = node.input[0]
    input2 = node.input[0]
    input1_dim = diction[input1]
    input2_dim = diction[input2]
    if (input1_dim.size >= input2_dim.size):
        out_size = np.append(input1_dim[0:-1], input2_dim[-1])
    else:
        out_size = np.append(input2_dim[0:-1], input1_dim[-1])
    input1_dim = np.array(input1_dim)
    input2_dim = np.array(input2_dim)
    macs = np.prod(input1_dim)/input1_dim[-1]*np.prod(input2_dim)
    output_name = diction[node.output[0]]
    return macs, out_size, output_name
#def count_onnx_
def counter_MatMul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(np.append(input_size[0:-1],output_size[-1]))
