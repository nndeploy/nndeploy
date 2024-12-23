import argparse
import onnx
from onnx import shape_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Modify input shapes of an ONNX model.')
    parser.add_argument('src_model_path', type=str, help='Path to the source ONNX model.')
    parser.add_argument('dst_model_path', type=str, help='Path to save the modified ONNX model.')
    parser.add_argument('--input_shapes', type=str, required=True, help='Input shapes in the format "input1:1,3,224,224;input2:1,10".')
    parser.add_argument('--infer_shape', action='store_true', help='Whether to infer shapes for the ONNX model.')
    return parser.parse_args()

def parse_input_shapes(input_shapes_str):
    input_shapes = {}
    for item in input_shapes_str.split(';'):
        name, shape_str = item.split(':')
        shape = list(map(int, shape_str.split(',')))
        input_shapes[name] = shape
    return input_shapes

def write_onnx_input_shape(src_model_path, dst_model_path, input_shapes, infer_shape):
    """
    修改 ONNX 模型的输入形状

    参数:
    src_model_path (str): 源 ONNX 模型文件路径
    dst_model_path (str): 保存修改后 ONNX 模型的路径
    input_shapes (dict): 输入形状字典，键为输入名称，值为形状列表
    infer_shape (bool): 是否推断 ONNX 模型的形状

    返回:
    None
    """
    model = onnx.load(src_model_path)
    graph = model.graph

    for input_tensor in graph.input:
        if input_tensor.name in input_shapes:
            shape = input_shapes[input_tensor.name]
            while len(input_tensor.type.tensor_type.shape.dim) > 0:
                input_tensor.type.tensor_type.shape.dim.pop()
            for dim in shape:
                new_dim = input_tensor.type.tensor_type.shape.dim.add()
                new_dim.dim_value = dim

    if infer_shape:
        model = shape_inference.infer_shapes(model)

    onnx.checker.check_model(model)
    onnx.save(model, dst_model_path)

if __name__ == '__main__':
    args = parse_args()
    src_model_path = args.src_model_path
    dst_model_path = args.dst_model_path
    input_shapes = parse_input_shapes(args.input_shapes)
    write_onnx_input_shape(src_model_path, dst_model_path, input_shapes, args.infer_shape)

# resnet
## python3 onnx_inputs_shape.py --src_model_path /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.onnx /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.staticshape.onnx --input_shapes data:1,3,224,224 --infer_shape
# RMBGV1.4.onnx
## python3 onnx_inputs_shape.py /home/ascenduserdg01/github/nndeploy/build/RMBGV1.4.onnx /home/ascenduserdg01/github/nndeploy/build/RMBGV1.4.staticshape.onnx --input_shapes input:1,3,1024,1024 --infer_shape

