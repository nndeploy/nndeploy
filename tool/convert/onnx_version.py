# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:05:03 2021

"""
# 


import argparse
import onnx
from onnx import version_converter


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX model to a different version.')
    parser.add_argument('--src_model_path', type=str, required=True, help='Path to the source ONNX model.')
    parser.add_argument('--target_ir_version', type=int, required=True, help='Target ONNX version to convert to.')
    parser.add_argument('--dst_model_path', type=str, default="", help='Path to save the converted ONNX model.')
    return parser.parse_args()

def onnx_version_converter(src_model_path, target_ir_version, dst_model_path=""):
    model = onnx.load(src_model_path)
    print(f"Current model version: {model.ir_version}")
    try:
        converted_model = version_converter.convert_version(model, target_ir_version)
        if dst_model_path == "":
            dst_model_path = src_model_path.replace(".onnx", f"_v{target_ir_version}.onnx") 
    except Exception as e:
        print(f"Error occurred during version conversion: {e}")
        return
    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, dst_model_path)


if __name__ == '__main__':
    args = parse_args()
    src_model_path = args.src_model_path
    target_ir_version = args.target_ir_version
    dst_model_path = args.dst_model_path
    onnx_version_converter(src_model_path, target_ir_version, dst_model_path)


# python3 onnx_version.py --src_model_path /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --target_version 10
    