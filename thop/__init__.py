from .utils import clever_format
from .profile import profile, profile_origin
# from .onnx_profile import OnnxProfile
import torch

default_dtype = torch.float64
from .__version__ import __version__