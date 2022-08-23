import torch
import torch.nn
import onnx
from onnx import numpy_helper
import numpy as np
from thop.vision.onnx_counter import onnx_operators


class OnnxProfile:
    def __init__(self):
        pass

    def calculate_params(self, model: onnx.ModelProto):
        onnx_weights = model.graph.initializer
        params = 0

        for onnx_w in onnx_weights:
            try:
                weight = numpy_helper.to_array(onnx_w)
                params += np.prod(weight.shape)
            except Exception as _:
                pass

        return params

    def create_dict(self, weight, input, output):
        diction = {}
        for w in weight:
            dim = np.array(w.dims)
            diction[str(w.name)] = dim
            if dim.size == 1:
                diction[str(w.name)] = np.append(1, dim)
        for i in input:
            # print(i.type.tensor_type.shape.dim[0].dim_value)
            dim = np.array(i.type.tensor_type.shape.dim[0].dim_value)
            # print(i.type.tensor_type.shape.dim.__sizeof__())
            # name2dims[str(i.name)] = [dim]
            dim = []
            for key in i.type.tensor_type.shape.dim:
                dim = np.append(dim, int(key.dim_value))
                # print(key.dim_value)
            # print(dim)
            diction[str(i.name)] = dim
            if dim.size == 1:
                diction[str(i.name)] = np.append(1, dim)
        for o in output:
            dim = np.array(o.type.tensor_type.shape.dim[0].dim_value)
            diction[str(o.name)] = [dim]
            if dim.size == 1:
                diction[str(o.name)] = np.append(1, dim)
        return diction

    def nodes_counter(self, diction, node):
        if node.op_type not in onnx_operators:
            print("Sorry, we haven't add ", node.op_type, "into dictionary.")
            return 0, None, None
        else:
            fn = onnx_operators[node.op_type]
            return fn(diction, node)

    def calculate_macs(self, model: onnx.ModelProto) -> torch.DoubleTensor:
        macs = 0
        name2dims = {}
        weight = model.graph.initializer
        nodes = model.graph.node
        input = model.graph.input
        output = model.graph.output
        name2dims = self.create_dict(weight, input, output)
        macs = 0
        for n in nodes:
            macs_adding, out_size, outname = self.nodes_counter(name2dims, n)

            name2dims[outname] = out_size
            macs += macs_adding
        return np.array(macs[0])
