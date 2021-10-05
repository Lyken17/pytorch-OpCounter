import onnx
from thop import onnx_profile
from onnx import numpy_helper
import numpy as np
model = onnx.load("conv.onnx")
print(onnx.helper.printable_graph(model.graph))
onnx_profile = onnx_profile()
print(onnx_profile.calculate_macs(model))
