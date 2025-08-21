# graph_runner.py

from __future__ import annotations
import json
import time
import logging
import traceback
from typing import Dict, Any, Tuple, List
import argparse

import nndeploy.base
import nndeploy.device
from .node import add_global_import_lib, import_global_import_lib
from .graph_runner import GraphRunner
     
       
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--name", type=str, default="", required=False)
    parser.add_argument("--task_id", type=str, default="", required=False)
    
    # image.jpg or decodec1:image.jpg
    parser.add_argument("--input_path", "-i", type=str, nargs='*', default=[], required=False)
    # image.jpg or decodec1:image.jpg
    parser.add_argument("--output_path", "-o", type=str, nargs='*', default=[], required=False)
    
    # node_name:param_key:param_value
    parser.add_argument("--node_param", "-np", type=str, nargs='*', default=[], required=False)
    
    # kParallelTypeSequential
    parser.add_argument("--parallel_type", type=str, default="", required=False)
    parser.add_argument("--dump", action="store_true", default=False, required=False)
    
    # plugin
    parser.add_argument("--plugin", type=str, nargs='*', default=[], required=False)
    return parser.parse_args()


class GraphRunnerArgs:
    def __init__(self):
        self.json_file = ""
        self.name = ""
        self.task_id = ""
        self.input_path = {}
        self.output_path = {}
        self.node_param = []
        self.parallel_type = ""
        self.dump = False
        self.plugin = []
        
    def parse_args(self):
        args = parse_args()
        self.json_file = args.json_file
        self.name = args.name
        self.task_id = args.task_id
        
        # 解析输入路径参数
        i = 0
        for item in args.input_path:
            if ":" in item:
                key, value = item.split(":", 1)
                self.input_path[key] = value
            else:
                self.input_path[str(i)] = item
            i += 1
        
        # 解析输出路径参数
        i = 0
        for item in args.output_path:
            if ":" in item:
                key, value = item.split(":", 1)
                self.output_path[key] = value
            else:
                self.output_path[str(i)] = item
            i += 1
        
        for item in args.node_param:
            self.node_param.append(item)
        
        for item in args.plugin:
            self.plugin.append(item)
        
        self.parallel_type = args.parallel_type
        self.dump = args.dump


def main():
    args = GraphRunnerArgs()
    args.parse_args()
    
    if args.plugin != []:
        for plugin_path in args.plugin:
            add_global_import_lib(plugin_path)
        import_global_import_lib()
    
    graph_json_str = ""
    with open(args.json_file, "r") as f:
        graph_json_str = f.read()
    gr = GraphRunner()
    time_profiler_map, results, _, _ = gr.run(graph_json_str, args.name, args.task_id, args)
    print(time_profiler_map)
    print(results)

if __name__ == "__main__":
    main()