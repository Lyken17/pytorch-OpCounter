# THOP: PyTorch-OpCounter

## How to install 
    
`pip install thop` (now continously intergrated on [Github actions](https://github.com/features/actions))

OR

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
    
## Results of Recent Models

The implementation are adapted from `torchvision`. Following results can be obtained using [benchmark/evaluate_famours_models.py](benchmark/evaluate_famous_models.py).

<p align="center">
<table>
<tr>
<td>

Model | Params(M) | MACs(G)
---|---|---
alexnet | 58.27 | 0.72
vgg11 | 126.71 | 7.21
vgg11_bn | 126.71 | 7.24
vgg13 | 126.88 | 10.66
vgg13_bn | 126.89 | 10.70
vgg16 | 131.95 | 14.54
vgg16_bn | 131.96 | 14.59
vgg19 | 137.01 | 18.41
vgg19_bn | 137.02 | 18.47
resnet18 | 11.15 | 1.70
resnet34 | 20.79 | 3.43
resnet50 | 24.37 | 3.85
resnet101 | 42.49 | 7.33
resnet152 | 57.40 | 10.81
wide_resnet101_2 | 121.01 | 21.27
wide_resnet50_2 | 65.69 | 10.67


</td>
<td>

Model | Params(M) | MACs(G)
---|---|---
resnext101_32x8d | 84.68 | 15.41
resnext50_32x4d | 23.87 | 4.00
densenet121 | 7.61 | 2.70
densenet161 | 27.35 | 7.31
densenet169 | 13.49 | 3.20
densenet201 | 19.09 | 4.09
squeezenet1_0 | 1.19 | 0.77
squeezenet1_1 | 1.18 | 0.33
mnasnet0_5 | 2.12 | 0.13
mnasnet0_75 | 3.02 | 0.23
mnasnet1_0 | 4.18 | 0.31
mnasnet1_3 | 5.99 | 0.49
mobilenet_v2 | 3.34 | 0.31
shufflenet_v2_x0_5 | 1.30 | 0.04
shufflenet_v2_x1_0 | 2.17 | 0.14
shufflenet_v2_x1_5 | 3.34 | 0.29
shufflenet_v2_x2_0 | 7.05 | 0.56

</td>
</tr>
</p>
