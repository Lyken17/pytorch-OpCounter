import torch 
import torch as th
import torch.nn as nn
from distutils.version import LooseVersion

if LooseVersion(torch.__version__) < LooseVersion("1.8.0"):
    logging.warning(
        f"torch.fx requires version higher than 1.8.0. "\
        f"But You are using an old version PyTorch {torch.__version__}. ")

def count_clamp(input_shapes, output_shapes):
    return 0

def count_mul(input_shapes, output_shapes):
    # element-wise
    return output_shapes[0].numel()

def count_nn_linear(input_shapes, output_shapes):
    in_shape = input_shapes[0]
    out_shape = output_shapes[0]
    in_features = in_shape[-1]
    num_elements  = out_shape.numel()
    return in_features * num_elements

count_map = {
    nn.Linear: count_nn_linear,
    "clamp": count_clamp,
    "<built-in function mul>": count_mul,
    "<built-in function truediv>": count_mul,
}

from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

def null_print(*args, **kwargs):
    return

def fx_profile(m: nn.Module, input: th.Tensor, verbose=True):
    gm : torch.fx.GraphModule = symbolic_trace(m)
    g = gm.graph
    ShapeProp(gm).propagate(input)

    fprint = null_print
    if verbose:
        fprint = print
    
    v_maps = {}
    total_flops = 0

    for node in gm.graph.nodes:
        # print(f"{node.target},\t{node.op},\t{node.meta['tensor_meta'].dtype},\t{node.meta['tensor_meta'].shape}")
        fprint(f"NodeOP:{node.op},\tTarget:{node.target},\tNodeName:{node.name},\tNodeArgs:{node.args}")
        node_op_type = str(node.target).split(".")[-1]
        node_flops = None
        
        input_shapes = [] 
        output_shapes = []
        fprint("input_shape:", end="\t")
        for arg in node.args:
            if str(arg) not in v_maps:
                continue
            fprint(f"{v_maps[str(arg)]}", end="\t")
            input_shapes.append(v_maps[str(arg)])
        fprint()
        fprint(f"output_shape:\t{node.meta['tensor_meta'].shape}")
        output_shapes.append(node.meta['tensor_meta'].shape)
        
        if node.op in ["output", "placeholder"]:
            node_flops = 0
        elif node.op == "call_function":
            # torch internal functions
            if str(node.target) in count_map:
                node_flops = count_map[str(node.target)](input_shapes, output_shapes)
            pass
        elif node.op == "call_method":
            # torch internal functions
            # print(str(node.target) in count_map, str(node.target), count_map.keys())
            if str(node.target) in count_map:
                node_flops = count_map[str(node.target)](input_shapes, output_shapes)
        elif node.op == "call_module":
            # torch.nn modules
            m = getattr(net, node.target, None)
            fprint(type(m), type(m) in count_map)
            if type(m) in count_map:
                node_flops = count_map[type(m)](input_shapes, output_shapes)
            if node_op_type not in ["relu", "maxpool", "avgpool"]:
                fprint(f"weight_shape: {net.state_dict()[node.target + '.weight'].shape}")
            else:
                fprint(f"weight_shape: None")
        
        v_maps[str(node.name)] =  node.meta['tensor_meta'].shape

        fprint(f"NodeFlops: {node_flops}")
        if node_flops is not None:
            total_flops += node_flops
        fprint("==" * 20)
    return total_flops




if __name__ == '__main__':
    class MyOP(nn.Module):
        def forward(self, input):
            return input / 1

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(5, 3)
            self.linear2 = torch.nn.Linear(5, 3)
            self.myop = MyOP()

        def forward(self, x):
            out1 = self.linear1(x)
            out2 = self.linear2(x).clamp(min=0.0, max=1.0)
            return self.myop(out1 + out2)
            
    net = MyModule()
    data = th.randn(20, 5)
    flops = fx_profile(net, data, verbose=False)
    print(flops)
    