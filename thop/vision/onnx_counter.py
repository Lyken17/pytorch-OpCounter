import torch
import numpy as np

from thop.vision.basic_hooks import zero_ops
from .counter import *


def onnx_counter_matmul(diction, node):
    input1 = node.input[0]
    input2 = node.input[1]
    input1_dim = diction[input1]
    input2_dim = diction[input2]
    out_size = np.append(input1_dim[0:-1], input2_dim[-1])
    output_name = node.output[0]
    macs = counter_matmul(input1_dim, out_size[-2:])
    return macs, out_size, output_name


def onnx_counter_add(diction, node):
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        out_size = diction[node.input[1]]
    else:
        out_size = diction[node.input[0]]
    output_name = node.output[0]
    macs = counter_zero_ops()
    return macs, out_size, output_name


def onnx_counter_conv(diction, node):
    # print(node)
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
    output_size = np.append(output_size, hw)
    # print(output_size)
    #print(np.prod(dim_bias), np.prod(dim_kernel), np.prod(output_size))
    macs = counter_conv(np.prod(dim_bias), np.prod(
        dim_kernel), np.prod(output_size))
    output_name = node.output[0]
    return macs, output_size, output_name


def onnx_counter_constant(diction, node):
    # print(node)
    macs = counter_zero_ops()
    output_name = node.output[0]
    output_size = [1]
    #print(macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_mul(diction, node):
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]]
    else:
        input_size = diction[node.input[0]]
    macs = counter_mul(np.prod(input_size))
    output_size = diction[node.input[0]]
    output_name = node.output[0]
    return macs, output_size, output_name


def onnx_counter_bn(diction, node):
    input_size = diction[node.input[0]]
    macs = counter_norm(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_relu(diction, node):
    input_size = diction[node.input[0]]
    macs = counter_relu(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_reducemean(diction, node):
    input_size = diction[node.input[0]]
    macs = counter_zero_ops()
    output_name = node.output[0]
    output_size = input_size
    #print("reduce",macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_sub(diction, node):
    input_size = diction[node.input[0]]
    macs = counter_zero_ops()
    output_name = node.output[0]
    output_size = input_size
    #print("sub",macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_pow(diction, node):
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]]
    else:
        input_size = diction[node.input[0]]
    macs = counter_pow(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    #print("pow",macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_sqrt(diction, node):
    input_size = diction[node.input[0]]
    macs = counter_sqrt(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    #print("sqrt",macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_div(diction, node):
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]]
    else:
        input_size = diction[node.input[0]]
    macs = counter_div(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    #print("div",macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_instance(diction, node):
    input_size = diction[node.input[0]]
    macs = counter_norm(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_softmax(diction, node):
    input_size = diction[node.input[0]]
    dim = node.attribute[0].i
    nfeatures = input_size[dim]
    batch_size = np.prod(input_size) / nfeatures
    macs = counter_softmax(nfeatures, batch_size)
    output_name = node.output[0]
    output_size = input_size
    #print("soft",macs, output_size, output_name)
    return macs, output_size, output_name


onnx_operators = {
    'MatMul': onnx_counter_matmul,
    'Add': onnx_counter_add,
    'Conv': onnx_counter_conv,
    'Mul': onnx_counter_mul,
    'Constant': onnx_counter_constant,
    'BatchNormalization': onnx_counter_bn,
    'Relu': onnx_counter_relu,
    'ReduceMean': onnx_counter_reducemean,
    'Sub': onnx_counter_sub,
    'Pow': onnx_counter_pow,
    'Sqrt': onnx_counter_sqrt,
    'Div': onnx_counter_div,
    'InstanceNormalization': onnx_counter_instance,
    'Softmax': onnx_counter_softmax,
    None: None,
}
