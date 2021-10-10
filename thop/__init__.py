from .utils import clever_format
from .profile import profile, profile_origin
import torch
from .jit_profile import JitProfile
default_dtype = torch.float64