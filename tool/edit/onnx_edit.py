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
    parser.add_argument('src_model_path', type=str, help='Path to the source ONNX model.')
    parser.add_argument('--dst_model_path', type=str, default="", help='Path to save the converted ONNX model.')
    parser.add_argument('--old_node_name', type=str, required=True, help='Old node name.')
    parser.add_argument('--new_node_name', type=str, required=True, help='New node name.')
    return parser.parse_args()

def onnx_edit_node_name(src_model_path, old_node_name, new_node_name, dst_model_path=""):
    model = onnx.load(src_model_path)
    for node in model.graph.node:
        if node.name == old_node_name:
            node.name = new_node_name
            break
    if dst_model_path == "":
        dst_model_path = src_model_path.replace(".onnx", f"_edit_node_name.onnx") 
    onnx.checker.check_model(model)
    onnx.save(model, dst_model_path)


if __name__ == '__main__':
    args = parse_args()
    src_model_path = args.src_model_path
    old_node_name = args.old_node_name
    new_node_name = args.new_node_name
    dst_model_path = args.dst_model_path
    onnx_edit_node_name(src_model_path, old_node_name, new_node_name, dst_model_path)


# python3 onnx_edit.py /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx --old_node_name "yolov8n.p1.conv" --new_node_name "yolov8n.p1.conv1"
    