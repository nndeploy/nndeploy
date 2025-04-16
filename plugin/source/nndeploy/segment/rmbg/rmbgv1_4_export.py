# 官方直接提供了onnx模型，下载连接: https://huggingface.co/briaai/RMBGV1.4/tree/main/onnx

# 在python环境直接导出为onnx模型进行优化，具体操作如下：


import onnx
import onnxsim

export_path = "/home/ascenduserdg01/github/nndeploy/build/RMBGV1.4.onnx"
sim_export_path = export_path.replace(".onnx", ".sim.onnx")
model = onnx.load(export_path)
model, check = onnxsim.simplify(model, overwrite_input_shapes={"input": [1, 3, 1024, 1024]})
onnx.save(model, sim_export_path)

print(f"Model optimized and saved to {sim_export_path}")


# 从onnx模型转换为om模型文件
# ```bash
# atc --model=/home/ascenduserdg01/github/nndeploy/build/RMBGV1.4.sim.onnx --output=/home/ascenduserdg01/github/nndeploy/build/RMBGV1.4.onnx.om --framework=5 --soc_version=Ascend910B4
# ```

# 从onnx模型转换为nndeploy自定义模型文件

# ```python
# python3 converter.py --model_value path/to/RMBGV1.4.sim.onnx --structure_file_path path/to/RMBGV1.4.slim.onnx.json --weight_file_path path/to/RMBGV1.4.slim.onnx.safetensors
# ``