__all__ = ['Node']


class Node:
    def __init__(self, operator, attributes, inputs, outputs, scope):
        self.operator = operator
        self.attributes = attributes
        self.inputs = inputs
        self.outputs = outputs
        self.scope = scope