__all__ = ['Graph']


class Graph:
    def __init__(self, name, variables, inputs, outputs, nodes):
        self.name = name
        self.variables = variables
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes