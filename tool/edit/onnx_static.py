# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:05:03 2021

"""
# 


import argparse
import onnx


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ONNX model operator statistics.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model.')
    return parser.parse_args()

def analyze_onnx_model(model_path):
    model = onnx.load(model_path)
    op_type_count = {}
    total_compute = 0
    total_memory_access = 0

    for node in model.graph.node:
        op_type = node.op_type
        if op_type not in op_type_count:
            op_type_count[op_type] = 0
        op_type_count[op_type] += 1

        # 假设每个节点的计算量和访存量可以通过某种方式计算
        # 这里使用伪代码表示
        compute = calculate_compute(node)
        memory_access = calculate_memory_access(node)

        total_compute += compute
        total_memory_access += memory_access

    print("算子种类数量:", len(op_type_count))
    for op_type, count in op_type_count.items():
        print(f"算子类型: {op_type}, 数量: {count}")
    print("总计算量:", total_compute)
    print("总访存量:", total_memory_access)

def calculate_compute(node):
    # 伪代码：根据节点信息计算计算量
    return -1  # 示例返回值

def calculate_memory_access(node):
    # 伪代码：根据节点信息计算访存量
    return -1  # 示例返回值

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    analyze_onnx_model(model_path)


# python3 onnx_static.py /home/ascenduserdg01/github/nndeploy/build/resnet50-v1-7.sim.onnx
    