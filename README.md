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
alexnet | 61.10 | 0.77
vgg11 | 132.86 | 7.74
vgg11_bn | 132.87 | 7.77
vgg13 | 133.05 | 11.44
vgg13_bn | 133.05 | 11.49
vgg16 | 138.36 | 15.61
vgg16_bn | 138.37 | 15.66
vgg19 | 143.67 | 19.77
vgg19_bn | 143.68 | 19.83
resnet18 | 11.69 | 1.82
resnet34 | 21.80 | 3.68
resnet50 | 25.56 | 4.14
resnet101 | 44.55 | 7.87
resnet152 | 60.19 | 11.61
wide_resnet101_2 | 126.89 | 22.84
wide_resnet50_2 | 68.88 | 11.46

</td>
<td>

Model | Params(M) | MACs(G)
---|---|---
resnext50_32x4d | 25.03 | 4.29
resnext101_32x8d | 88.79 | 16.54
densenet121 | 7.98 | 2.90
densenet161 | 28.68 | 7.85
densenet169 | 14.15 | 3.44
densenet201 | 20.01 | 4.39
squeezenet1_0 | 1.25 | 0.82
squeezenet1_1 | 1.24 | 0.35
mnasnet0_5 | 2.22 | 0.14
mnasnet0_75 | 3.17 | 0.24
mnasnet1_0 | 4.38 | 0.34
mnasnet1_3 | 6.28 | 0.53
mobilenet_v2 | 3.50 | 0.33
shufflenet_v2_x0_5 | 1.37 | 0.05
shufflenet_v2_x1_0 | 2.28 | 0.15
shufflenet_v2_x1_5 | 3.50 | 0.31
shufflenet_v2_x2_0 | 7.39 | 0.60
inception_v3 | 27.16 | 5.75

</td>
</tr>
</p>
