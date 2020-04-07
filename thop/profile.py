from distutils.version import LooseVersion

from thop.vision.basic_hooks import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logger.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=torch.__version__))

default_dtype = torch.float64

register_hooks = {
    nn.ZeroPad2d: zero_ops, # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample
}


def profile(model, inputs, custom_ops=None, verbose=True):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logger.warning("Either .total_ops or .total_params is already defined in %s. "
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1, dtype=default_dtype))
        m.register_buffer('total_params', torch.zeros(1, dtype=default_dtype))

        for p in m.parameters():
            m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            print("[Customized] %s" % m._get_name())
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            print("[THOP Defined] %s" % m._get_name())
            fn = register_hooks[m_type]
        else:
            print("[Not found] %s" % m._get_name())

        if fn is None:
            if verbose:
                logger.info("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                logger.info("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

    return total_ops, total_params


def profile_2(model: nn.Module, inputs, custom_ops=None, verbose=True):
    handler_collection = {}
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        # if hasattr(m, "total_ops") or hasattr(m, "total_params"):
        #     logger.warning("Either .total_ops or .total_params is already defined in %s. "
        #                    "Be careful, it might change your code's behavior." % m._get_name())
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        # if defined both op maps, custom_ops takes higher priority.
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            print("[Customized] %s" % m._get_name())
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            print("[THOP Defined] %s" % m._get_name())
            fn = register_hooks[m_type]
        else:
            print("[Not found] %s" % m._get_name())

        # if verbose:
        #     if fn is None:
        #         print("THOP has not implemented counting method for %s." % m._get_name())
        #     else:
        #        print("Register FLOP counter for module %s." % m._get_name())
        if fn is not None:
            handler_collection[m] = m.register_forward_hook(fn)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops, m.total_params
            else:
                m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params

    total_ops, total_params = dfs_count(model)

    # reset model to original status
    model.train(training)
    for m, handler in handler_collection.items():
        handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params
