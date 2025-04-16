
# 
# The following commands are for installing necessary Python libraries to handle and optimize ONNX models
# pip3 install onnx
# pip3 install onnxsim
# pip3 install ultralytics
#

from ultralytics import YOLO  # Import YOLO model class from ultralytics library
import onnx  # Import ONNX library for handling ONNX models
import onnxsim  # Import ONNXSIM library for simplifying ONNX models
from onnx import version_converter

# Load YOLO model, specifying the model file as "yolo11s.pt"
# This repository directly supports exporting to ONNX format
model = YOLO("yolo11s.pt")
# export_path = model.export(format="onnx", simplify=False)
export_path = model.export(format="onnx", simplify=False, opset=11)
print(f"Model exported to {export_path}")

# Use onnxsim library to simplify the model, removing unnecessary computations and parameters
sim_export_path = export_path.replace(".onnx", ".sim.onnx")
model = onnx.load(export_path)
model, check = onnxsim.simplify(model)
onnx.save(model, sim_export_path)
print(f"Model optimized and saved to {sim_export_path}")

# Use Huawei's ATC tool to convert the ONNX model to an OM model, suitable for Ascend hardware
# note：atc 工具要求numpy版本必须低于2.0(AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead..)
# 
#  ```bash
#  atc --model=yolo11s.onnx --output=yolo11s.sim.onnx.om --framework=5 --soc_version=Ascend910B4
#  atc --model=/home/ascenduserdg01/github/nndeploy/build/yolo11s.sim.onnx --output=/home/ascenduserdg01/github/nndeploy/build/yolo11s.sim.onnx.om --framework=5 --soc_version=Ascend910B4
#  ```
