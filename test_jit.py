import torch
from torchvision.models import resnet18
from torchprofile import profile_macs
import torch.nn as nn
from thop import JitProfile
input = torch.rand(1,10)
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20,10)
    def forward(self,input1):

        lin1 = self.linear1(input1)
        lin2 = self.linear2(lin1)
        return lin2
linear = model1()
linear_jit = torch.jit.script(linear)
linear_trace = torch.jit.trace(linear,input)
#print(JitProfile.calculate_params(linear))
model1 = resnet18()
input1 = torch.rand(1,3,224,224)
print(JitProfile.calculate_macs(model1,input1))
