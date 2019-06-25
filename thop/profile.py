import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .count_hooks import *

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose2d: count_convtranspose2d,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,

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


def profile(model, inputs, custom_ops={}, verbose=True):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            raise Warning("Either .total_ops or .total_params is already defined in %s.\n"
                          "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    # original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params
