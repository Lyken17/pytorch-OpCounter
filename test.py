import torch
import thop
import torchvision
dummy_input = torch.randn(10, 3, 224, 224)
model = torchvision.models.alexnet(pretrained=True)

flops = thop.profile(model, inputs=(dummy_input,), verbose=True)
print(flops)
