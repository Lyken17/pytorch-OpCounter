__all__ = ['Variable']


class Variable:
    def __init__(self, name, dtype, shape=None):
        self.name = name
        self.dtype = dtype
        self.shape = shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype.lower()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def ndim(self):
        return len(self.shape)

    def size(self):
        return self.shape

    def dim(self):
        return self.ndim