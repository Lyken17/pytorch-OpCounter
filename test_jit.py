import torch
from torchvision.models import vgg11
from thop import JitProfile
model1 = vgg11()
input1 = torch.rand(1,3,224,224)
print(JitProfile.calculate_macs(model1,input1))
