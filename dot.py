from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ('x').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    from collections import defaultdict
    info_nodes = {}
    tail2head_edges = defaultdict(set)
    head2tail_edges = defaultdict(set)
    lazy_node_registration = []
    lazy_edge_registration = []

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                nid = str(id(var))
                name = size_to_str(var.size())
                kwargs = {
                        "fillcolor": "orange"
                }
                dot.node(nid, name, fillcolor='orange')
                info_nodes[str(id(var))] = {
                    "name": str(id(var)),
                    "size": var.size(),
                    "kwargs": kwargs
                }
            elif hasattr(var, 'variable'):
                # variable node
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                nid = str(id(var))
                dot.node(nid, node_name, fillcolor='lightblue')
                info_nodes[str(id(var))] = {
                    "name": node_name,
                    "size": u.size()
                }
            elif var in output_nodes:
                # final output
                nid = str(id(var))
                node_name = str(type(var).__name__)
                dot.node(str(id(var)), node_name, fillcolor='darkolivegreen1')
                info_nodes[str(id(var))] = {
                    "name": str(type(var).__name__),
                    "size": None
                }
            else:
                # other output(s)
                dot.node(str(id(var)), str(type(var).__name__))
                info_nodes[str(id(var))] = {
                    "name": str(type(var).__name__),
                    "size": None
                }

            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        tail_name = str(id(u[0]))
                        head_name = str(id(var))
                        # dot.edge(tail_name, head_name, )
                        tail2head_edges[tail_name].add(head_name)
                        head2tail_edges[head_name].add(tail_name)
                        add_nodes(u[0])

            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    tail_name = str(id(t))
                    head_name = str(id(var))
                    # dot.edge(tail_name, head_name)
                    tail2head_edges[tail_name].add(head_name)
                    head2tail_edges[head_name].add(tail_name)
                    add_nodes(t)

    # handle multiple outputs`
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    for key, value in info_nodes.items():
        in_edges = len(head2tail_edges[key])
        dot.node(key, value["name"] + "\tid:%s" % key +"\t in_edges:%d" % in_edges)

    # topologic sort to infer the shape
    from queue import Queue
    zero_edges = Queue()
    edges_count = {}
    for key, value in info_nodes.items():
        in_edges = len(head2tail_edges[key])
        edges_count[key] = in_edges
        if in_edges == 0:
            zero_edges.put(key)

    while not zero_edges.empty():
        tail = zero_edges.get()
        print(tail)
        for head in tail2head_edges[tail]:
            dot.edge(tail, head, label=size_to_str(info_nodes[tail]["size"]))
            # define by op here
            if "inputs" not in info_nodes[head]:
                info_nodes[head]["inputs"] = list()
            info_nodes[head]["inputs"].append(info_nodes[tail]["size"])
            edges_count[head] -= 1
            if edges_count[head] == 0:
                # all inputs ready
                input_list = info_nodes[head]["inputs"]
                node_name = info_nodes[head]["name"]
                d_size = torch.Size([-1])
                if "MulBackward" in node_name:
                    dsize = input_list[0]
                elif "AddBackward" in node_name:
                    d_size = input_list[0]
                elif "TBackward" in node_name:
                    d_size = input_list[0][::-1]
                elif "ThresholdBackward" in node_name:
                    d_size = input_list[0]
                elif "MmBackward" in node_name:
                    s1, s2 = input_list
                    if s1[1] == s2[0]:
                        d_size = torch.Size([s1[0], s2[1]])
                    else:
                        d_size = torch.Size([s2[0], s1[1]])

                info_nodes[head]["size"] = d_size
                zero_edges.put(head)


    resize_graph(dot)

    return dot, info_nodes, tail2head_edges, head2tail_edges


# For traces

def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': n.kind(),
                             'inputs': inputs,
                             'attr': attrs}))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type())}))

    return nodes


def make_dot_from_trace(trace):
    """ Produces graphs of torch.jit.trace outputs

    Example:
    >>> trace, = torch.jit.trace(model, args=(x,))
    >>> dot = make_dot_from_trace(trace)
    """
    # from tensorboardX
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    list_of_nodes = parse(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
