import onnx
from thop import OnnxProfile
from onnx import numpy_helper
import numpy as np
model = onnx.load("conv.onnx")
print(onnx.helper.printable_graph(model.graph))
onnx_profile = OnnxProfile()
print(onnx_profile.calculate_macs(model))
