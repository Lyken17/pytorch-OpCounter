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
    
* Improve the output readability

    Call `thop.clever_format` to give a better format of the output.
    ```python
    from thop import clever_format
    flops, params = clever_format([flops, params], "%.3f")
    ```    
    
## Results on Recent Models

<p align="center">
<table>
<tr>
<td>

Model | Params(M) | FLOPs(G)
---|---|---
alexnet | 58.27 | 0.72
densenet121 | 7.61 | 2.70
densenet161 | 27.35 | 7.31
densenet169 | 13.49 | 3.20
densenet201 | 19.09 | 4.09
resnet18 | 11.15 | 1.70
resnet34 | 20.79 | 3.43
resnet50 | 24.37 | 3.85
resnet101 | 42.49 | 7.33
resnet152 | 57.40 | 10.81

</td>
<td>

Model | Params(M) | FLOPs(G)
---|---|---
squeezenet1_0 | 1.19 | 0.77
squeezenet1_1 | 1.18 | 0.33
vgg11 | 126.71 | 7.21
vgg11_bn | 126.71 | 7.24
vgg13 | 126.88 | 10.66
vgg13_bn | 126.89 | 10.70
vgg16 | 131.95 | 14.54
vgg16_bn | 131.96 | 14.59
vgg19 | 137.01 | 18.41
vgg19_bn | 137.02 | 18.47

</td>
</tr>
</p>