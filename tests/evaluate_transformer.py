
import torch.nn as nn
from thop import profile
import torch

src=torch.rand((10, 32, 10))
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.linear1 = nn.Linear(10,512)
        self.linear2 = nn.Linear(10,512)
        self.transform = nn.Transformer()
    def forward(self,input):
        input1 = self.linear1(input)
        input2 = self.linear2(input)
        output = self.transform(input1,input2)
        return output
model2 = nn.Sequential(model1())
macs, params = profile(model2, inputs=(src, ))
print(macs,params)

