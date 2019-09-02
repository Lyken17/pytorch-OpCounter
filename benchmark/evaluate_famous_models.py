
from torchvision import models

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") and not "inception" in name
                     and callable(models.__dict__[name]))

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")

from thop.profile import profile
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
for name in model_names:
    model = models.__dict__[name]().to(device)
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    total_ops, total_params = profile(model, (inputs, ), verbose=False)
    print("%s | %.2f | %.2f" % (name, total_params / (1024 ** 2), total_ops / (1024 ** 3)))
