
import torch.nn as nn
from thop import profile
import torch

src=torch.rand((1, 1, 10))# S,N,x
class Model_transformer(nn.Module):
    def __init__(self):
        super(Model_transformer, self).__init__()
        self.linear1 = nn.Linear(10,512)
        self.linear2 = nn.Linear(10,512)
        self.transform = nn.Transformer(d_model = 512 ,nhead = 8, num_encoder_layers = 6)
    def forward(self,input):
        input1 = self.linear1(input)
        input2 = self.linear2(input)
        output = self.transform(input1,input2)
        return output
model = Model_transformer()
macs, params = profile(model, inputs=(src, ))
print(macs,params)

