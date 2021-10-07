import torch
import thop
import torchvision
dummy_input = torch.randn(1, 3, 224, 224)
model = torchvision.models.mnasnet1_3()

flops,params = thop.profile(model, inputs=(dummy_input,), verbose=True)
print(flops,params)
