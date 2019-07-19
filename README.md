# THOP: PyTorch-OpCounter

## How to install 
* Through PyPi
    
    `pip install thop`
    
* Using GitHub (always latest)
    
    `pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git`
    
## How to use 
* Basic usage 
    ```python
    from torchvision.models import resnet50
    from thop import profile
    model = resnet50()
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, ))
    ```    

* Define the rule for 3rd party module.
    
    ```python
    class YourModule(nn.Module):
        # your definition
    def count_your_model(model, x, y):
        # your rule here
    
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, ), 
                            custom_ops={YourModule: count_your_model})
    ```
    
## Results on Recent Models

<p align="center">
<table>
<tr>
<td>

Model | Params(M) | FLOPs(G)
---|---|---
alexnet | 58.27 | 0.77
densenet121 | 7.61 | 2.71
densenet161 | 27.35 | 7.34
densenet169 | 13.49 | 3.22
densenet201 | 19.09 | 4.11
resnet101 | 42.49 | 7.34
resnet152 | 57.40 | 10.82
resnet18 | 11.15 | 1.70
resnet34 | 20.79 | 3.43
resnet50 | 24.37 | 3.86

</td>
<td>

Model | Params(M) | FLOPs(G)
---|---|---
squeezenet1_0 | 1.19 | 1.01
squeezenet1_1 | 1.18 | 0.48
vgg11 | 126.71 | 7.98
vgg11_bn | 126.71 | 8.01
vgg13 | 126.88 | 11.82
vgg13_bn | 126.89 | 11.86
vgg16 | 131.95 | 16.12
vgg16_bn | 131.96 | 16.17
vgg19 | 137.01 | 20.43
vgg19_bn | 137.02 | 20.49

</td>
</tr>
</p>
