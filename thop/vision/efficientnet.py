import argparse
import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from efficientnet_pytorch.utils import Conv2dDynamicSamePadding, Conv2dStaticSamePadding
register_hooks = {

}