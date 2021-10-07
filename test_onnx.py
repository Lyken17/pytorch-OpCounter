import onnx
from thop import OnnxProfile
from onnx import numpy_helper
import numpy as np
model = onnx.load("model.onnx")# put the path of model you want to count
#print(onnx.helper.printable_graph(model.graph))
onnx_profile = OnnxProfile()
print(onnx_profile.calculate_macs(model))
print(onnx_profile.calculate_params(model))
