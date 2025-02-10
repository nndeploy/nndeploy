
# 
# pip install onnx
# pip install onnxruntime
# pip install onnxsim
# pip install ultralytics
#

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11s.pt")

# Export the model to ONNX format
export_path = model.export(format="onnx")

print(f"Model exported to {export_path}")

# 使用onnxsim优化模型
import onnx
import onnxsim

sim_export_path = export_path.replace(".onnx", "_sim.onnx")
model = onnx.load(export_path)
model, check = onnxsim.simplify(model)
onnx.save(model, sim_export_path)

print(f"Model optimized and saved to {sim_export_path}")
