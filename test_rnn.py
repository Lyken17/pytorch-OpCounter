import torch
from torchvision.models import resnet18
from torchprofile import profile_macs
import torch.nn as nn
input = torch.rand(1,10)
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.linear1 = nn.Linear(10, 20)
    def forward(self,input1):

        output = self.linear1(input1)
        
        return output
linear = model1()
linear_jit = torch.jit.script(linear)
linear_trace = torch.jit.trace(linear,input)
for i in linear_jit.parameters():
    print(i)
