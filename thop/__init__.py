from .utils import clever_format
from .profile import profile, profile_origin
from .onnx_profile import onnx_profile
import torch
default_dtype = torch.float64