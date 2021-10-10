import numpy as np
__all__ = ['handlers']
from thop.vision.counter import counter_mul, counter_addmm,\
    counter_addmv,counter_bmm,counter_matmul


def addmm(node):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape

    return counter_addmm(node.inputs[1].shape, node.inputs[2].shape)


def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    return counter_addmv(node.inputs[1].shape)


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return counter_bmm(node.inputs[0].shape,node.inputs[1].shape)


def matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return counter_mul(n)
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        # n, m = node.inputs[1].shape
        # return n * m
        return counter_mul(np.prod(node.inputs[1].shape))
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        # n, m = node.inputs[0].shape
        # return n * m
        return counter_mul(np.prod(node.inputs[0].shape))
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        # n, m = node.inputs[0].shape
        # m, p = node.inputs[1].shape
        # return n * m * p
        return counter_matmul(node.inputs[0].shape,node.inputs[1].shape)
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        # *b, n, m = node.inputs[1].shape
        # return np.prod(b) * n * m
        return counter_mul(np.prod(node.inputs[1].shape))
    elif node.inputs[1].ndim == 1:
        # # [..., n] = aten::matmul([..., n, m], [m])
        # *b, n, m = node.inputs[0].shape
        return counter_mul(np.prod(node.inputs[0].shape))
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return np.prod(b) * n * m * p


def mul(node):
    os = node.outputs[0].shape
    return np.prod(os)


def convolution(node):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        oc, ic, *ks = node.inputs[1].shape
    else:
        ic, oc, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    return np.prod(os) * ic * np.prod(ks)


def norm(node):
    if node.operator in ['aten::batch_norm', 'aten::instance_norm']:
        affine = node.inputs[1].shape is not None
    elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.operator)

    os = node.outputs[0].shape
    return np.prod(os) if affine else 0


def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    return np.prod(os)


def leaky_relu(node):
    os = node.outputs[0].shape
    return np.prod(os)


def upsample_bilinear2d(node):
    os = node.outputs[0].shape
    return np.prod(os) * 4


handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    (('aten::linear', 'aten::matmul'), matmul),
    (('aten::mul', 'aten::mul_'), mul),
    ('aten::_convolution', convolution),
    (('aten::batch_norm', 'aten::instance_norm', 'aten::layer_norm',
      'aten::group_norm'), norm),
    (('aten::adaptive_avg_pool1d', 'aten::adaptive_avg_pool2d',
      'aten::adaptive_avg_pool3d', 'aten::avg_pool1d', 'aten::avg_pool2d',
      'aten::avg_pool3d', 'aten::mean'), avg_pool_or_mean),
    ('aten::leaky_relu', leaky_relu),
    ('aten::upsample_bilinear2d', upsample_bilinear2d),
    (('aten::adaptive_max_pool1d', 'aten::adaptive_max_pool2d',
      'aten::adaptive_max_pool3d', 'aten::add', 'aten::add_',
      'aten::alpha_dropout', 'aten::cat', 'aten::chunk', 'aten::clamp',
      'aten::clone', 'aten::constant_pad_nd', 'aten::contiguous',
      'aten::detach', 'aten::div', 'aten::div_', 'aten::dropout',
      'aten::dropout_', 'aten::embedding', 'aten::eq', 'aten::feature_dropout',
      'aten::flatten', 'aten::floor', 'aten::floor_divide', 'aten::gt',
      'aten::hardtanh_', 'aten::hardtanh', 'aten::index', 'aten::int',  'aten::log_softmax',
      'aten::lt', 'aten::max_pool1d', 'aten::max_pool1d_with_indices',
      'aten::max_pool2d', 'aten::max_pool2d_with_indices', 'aten::max_pool3d',
      'aten::max_pool3d_with_indices', 'aten::max_unpool1d',
      'aten::max_unpool2d', 'aten::max_unpool3d', 'aten::ne',
      'aten::reflection_pad1d', 'aten::reflection_pad2d',
      'aten::reflection_pad3d', 'aten::relu', 'aten::relu_',
      'aten::replication_pad1d', 'aten::replication_pad2d',
      'aten::replication_pad3d', 'aten::rsub', 'aten::select', 'aten::sigmoid',
      'aten::size', 'aten::slice', 'aten::softmax', 'aten::softshrink',
      'aten::squeeze', 'aten::stack', 'aten::sub', 'aten::sum', 'aten::t',
      'aten::tanh', 'aten::threshold', 'aten::to', 'aten::transpose',
      'aten::upsample_nearest2d', 'aten::view', 'aten::zeros',
      'prim::constant', 'prim::listconstruct', 'prim::listunpack',
      'prim::numtotensor', 'prim::tupleconstruct'), None),
)
