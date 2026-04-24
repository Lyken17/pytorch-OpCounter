# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "2.0.19"

import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64

__all__ = ["clever_format", "default_dtype", "profile", "profile_origin"]
