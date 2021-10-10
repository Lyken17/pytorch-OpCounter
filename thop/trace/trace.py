import torch
import torch.jit
from ..trace.type import Variable,Node,Graph
def trace(model, args = ()):
    graph, _ = torch.jit._get_trace_graph(model,args)
    variables = {}
    #print(graph.__dir__())
    #print(graph.inputs)
    #print(graph)
    for x in graph.nodes():
        #print(x)
        for v in list(x.inputs()) or list(x.outputs()):
            if 'tensor' in v.type().kind().lower():
               if 'tensor' in v.type().kind().lower():
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=v.type().scalarType(),
                    shape=v.type().sizes(),
                )
            else:
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=str(v.type()),
                )
    pass
    nodes = []
    for x in graph.nodes():
        node = Node(
            operator=x.kind(),
            attributes={
                s: getattr(x, x.kindOf(s))(s)
                for s in x.attributeNames()
            },
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=x.scopeName() \
                .replace('Flatten/', '', 1) \
                .replace('Flatten', '', 1),
        )
        nodes.append(node)
    graph = Graph(
        name=model.__class__.__module__ + '.' + model.__class__.__name__,
        variables=[v for v in variables.values()],
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )
    return graph
    