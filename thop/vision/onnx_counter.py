import torch
import numpy as np

from thop.vision.basic_hooks import zero_ops
from .counter import *


def onnx_counter_MatMul(diction, node):
    input1 = node.input[0]
    input2 = node.input[1]
    input1_dim = diction[input1]
    input2_dim = diction[input2]
    out_size = np.append(input1_dim[0:-1], input2_dim[-1])
    output_name = node.output[0]
    macs = counter_MatMul(input1_dim, out_size)
    return macs, out_size, output_name


def onnx_counter_Add(diction, node):
    out_size = diction[node.input[1]]
    output_name = node.output[0]
    macs = counter_zero_ops()
    return macs, out_size, output_name


def onnx_counter_Conv(diction, node):
    #print(node)
    # bias,kernelsize,outputsize
    for i in node.input:
        if('bias' in i):
            dim_bias = diction[i]
        if('weight' in i):
            dim_weight = diction[i]  # cout, cin,kw,kh
    # print(dim_weight,dim_bias)
    for attr in node.attribute:
        # print(attr)
        if(attr.name == 'kernel_shape'):
            dim_kernel = attr.ints  # kw,kh
        if(attr.name == 'strides'):
            dim_stride = attr.ints
        if(attr.name == 'pads'):
            dim_pad = attr.ints
        if(attr.name == 'dilations'):
            dim_dil = attr.ints
            # print(dim_dil)
    dim_input = diction[node.input[0]]
    output_size = np.append(
        dim_input[0:-np.array(dim_kernel).size-1], dim_weight[0])
    hw = dim_input[-np.array(dim_kernel).size:]
    for i in range(hw.size):
        hw[i] = int((hw[i]+2*dim_pad[i]-dim_dil[i] *
        (dim_kernel[i]-1))/dim_stride[i])
    output_size = np.append(output_size,hw)
    #print(output_size)
    #print(np.prod(dim_bias), np.prod(dim_kernel), np.prod(output_size))
    macs = counter_conv(np.prod(dim_bias), np.prod(dim_kernel), np.prod(output_size))
    output_name = node.output[0]
    return macs, output_size, output_name

def onnx_counter_Constant(diction,node):
    #print(node)
    macs = counter_zero_ops()
    output_name = node.output[0]
    output_size = [1]
    print(macs, output_size, output_name)
    return macs, output_size, output_name

def onnx_counter_Mul(diction, node):
    print(node)
    
    pass


onnx_operators = {
    'MatMul': onnx_counter_MatMul,
    'Add': onnx_counter_Add,
    'Conv': onnx_counter_Conv,
    'Mul' : onnx_counter_Mul,
    'Constant' : onnx_counter_Constant,
    None: None,
}
