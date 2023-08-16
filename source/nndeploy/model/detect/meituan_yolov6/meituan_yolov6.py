#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx
import onnxoptimizer
import onnxsim

class MeituanYOLOV6:
    def __init__(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--src_path', type=str, default='./yolov6s.pt', help='src_path path')
        parser.add_argument('--dst_path', type=str, default='./yolov6s.pt', help='dst_path path')
        parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size, the order is: height width')  # height, width
        parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
        parser.add_argument('--batch-size', type=int, default=1, help='batch size')
        parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
        parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
        parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
        parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
        parser.add_argument('--iou-thres', type=float, default=0.65, help='iou threshold for NMS')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='conf threshold for NMS')
        args = parser.parse_args() 
    
    def export_onnx(self) -> None:
        pass
        

def main():
    MeituanYOLOV6()
    pass


if __name__ == "__main__":
    main()