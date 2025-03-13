
import argparse
from typing import List

import nndeploy
import nndeploy.base
import nndeploy._nndeploy_internal as _C


from .interpret import Interpret, create_interpret


# python3 nndeploy/ir/converter.py


class Convert():
    def __init__(self, type: str) -> None:
        self.interpret = create_interpret(nndeploy.base.name_to_model_type(type))

    def convert(self, model_value: List[str], structure_file_path: str, weight_file_path: str, input: List[_C.ir.ValueDesc] = []) -> None:
        self.interpret.interpret(model_value, input)
        self.interpret.save_model_to_file(structure_file_path, weight_file_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert model to nndeploy model.')
    parser.add_argument('--model_type', type=str, default="onnx", help='src model type.')
    parser.add_argument('--model_value', type=str, help='src model value.')
    parser.add_argument('--structure_file_path', type=str, default="", help='Path to save the converted model.')
    parser.add_argument('--weight_file_path', type=str, default="", help='Path to save the converted model.')
    parser.add_argument('--input', type=str, default="", help='Description of input tensors, format: name,type,shape;name,type,shape, where type and shape are optional')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 处理参数
    if args.model_value == "":
        print("model_value is required")
        exit(1)
    else:
        model_value = args.model_value.split(";")

    if args.structure_file_path == "":
        structure_file_path = model_value[0] + ".json"
    else:
        structure_file_path = args.structure_file_path  

    if args.weight_file_path == "":
        weight_file_path = model_value[0] + ".safetensors"
    else:
        weight_file_path = args.weight_file_path

    if args.input != "":
        input_list = args.input.split(";")
        input_list = [input.split(",") for input in input_list]
        input_list = [_C.ir.ValueDesc(name=input[0], type=input[1], shape=input[2]) for input in input_list]
    else:
        input_list = []

    # 转换模型
    convert = Convert(args.model_type)
    convert.convert(model_value, structure_file_path, weight_file_path, input_list)
