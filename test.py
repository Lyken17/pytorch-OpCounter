import torch
import thop

m = torch.nn.Conv2d(128, 128, 1)
x = torch.randn(1, 128, 16, 16)


flops = thop.profile(m, inputs=(x,), verbose=True)
fprint(flops)

