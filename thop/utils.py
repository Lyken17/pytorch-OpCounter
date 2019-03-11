import logging

import torch
import torch.nn as nn
from .count_hooks import *

register_hooks = {
    nn.Conv2d: count_conv2d,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,

    nn.MaxPool1d: count_maxpool,
    nn.MaxPool2d: count_maxpool,
    nn.MaxPool3d: count_maxpool,
    nn.AdaptiveMaxPool1d: count_adap_maxpool,
    nn.AdaptiveMaxPool2d: count_adap_maxpool,
    nn.AdaptiveMaxPool3d: count_adap_maxpool,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: None,
}


def profile(model, input_size, custom_ops={}, device="cpu"):
    handler_collection = []
    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            print("Not implemented for ", m)

        if fn is not None:
            print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval().to(device)
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params
