import torch
from torchvision import models
from thop.profile import profile

from pypapi import events, papi_high as high

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__") # and "inception" in name
                     and callable(models.__dict__[name]))

print("%s | %s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)", "PMU-FLOPS(G)"))
print("---|---|---|---")

device = "cpu"
torch.set_num_threads(1)

for name in model_names:
    model = models.__dict__[name]().double().to(device)
    dsize = (1, 3, 224, 224)
    if "inception" in name:
        dsize = (1, 3, 299, 299)
    inputs = torch.randn(dsize, dtype=torch.double).to(device)
    high.start_counters([events.PAPI_DP_OPS,])
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    flopsPMU=high.stop_counters()
    print("%s | %.2f | %.2f | %.2f" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3), flopsPMU[0] / 1e9 / 2))
