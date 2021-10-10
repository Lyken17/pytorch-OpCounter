import torch
import numpy as np
from .trace.trace import trace
from thop.vision.jit_handler import handlers


class JitProfile():
    def calculate_params(model):
        script_model = torch.jit.script(model)
        params = 0
        for param in script_model.parameters():
            params += np.prod(param.size())
            print(param)
        return params

    def calculate_macs(model, args=(),reduction = sum):
        results = dict()
        graph = trace(model, args)
        for node in graph.nodes:
            for operators, func in handlers:
                if isinstance(operators, str):
                    operators = [operators]
                if node.operator in operators:
                    if func is not None:
                        results[node] = func(node)
                        break

        if reduction is not None:
            return reduction(results.values())
        else:
            return results
