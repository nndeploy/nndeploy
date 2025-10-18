
import json
import time
import logging
import traceback
from typing import Dict, Any, Tuple, List
import argparse

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    # json_file
    parser.add_argument("--json_file", type=str, required=True)
    # node_param: node_name:param_key:param_value
    parser.add_argument("--node_param", "-np", type=str, nargs='*', default=[], required=False)
    # parallel_type: kParallelTypeSequential
    parser.add_argument("--parallel_type", type=str, default="Sequential", required=False)
    # remove_in_out_node: True
    parser.add_argument("--input_path", type=str, default="", required=False)
    parser.add_argument("--output_path", type=str, default="", required=False)
    parser.add_argument("--remove_in_out_node", action="store_true", default=False, required=False)
    args = parser.parse_args()
    json_file = args.json_file
    node_param_map = {
      args.node_param[i].split(':')[0]: {args.node_param[i].split(':')[1]: args.node_param[i].split(':')[2]} for i in range(len(args.node_param))
    }
    parallel_type = nndeploy.base.name_to_parallel_type[args.parallel_type]
    remove_in_out_node = args.remove_in_out_node
    input_path = args.input_path
    output_path = args.output_path
    return json_file, node_param_map, parallel_type, remove_in_out_node, input_path, output_path

def run_json(json_file, node_param_map, parallel_type):
    graph = nndeploy.dag.Graph(json_file)
    graph.set_node_value(node_param_map)
    graph.set_time_profile_flag(True)
    graph.set_parallel_type(parallel_type)
    graph.load_file(json_file)
    
    graph.init()
    
    count = graph.get_loop_count()
    for i in range(count):
        graph.run()
        if parallel_type != nndeploy.base.ParallelType.Pipeline:
            output = graph.get_output(0)
            result = output.get_graph_output()
            if result is not None:
                pass
    if parallel_type == nndeploy.base.ParallelType.Pipeline:
        for i in range(count):
            output = graph.get_output(0)
            result = output.get_graph_output()
            if result is not None:
                pass

    graph.deinit()
    
    nndeploy.base.time_profiler_print(graph.get_name())
  
  
def run_json_remove_in_out_node(json_file, node_param_map, parallel_type, input_path, output_path):
    graph = nndeploy.dag.Graph("")
    graph.set_node_value(node_param_map)
    graph.set_time_profile_flag(True)
    graph.set_parallel_type(parallel_type)
    graph.remove_in_out_node()
    graph.load_file(json_file)
    graph.init()
    
    input = graph.get_input(0)    
    import cv2
    image = cv2.imread(input_path)
    input.set(image)
    status = graph.run()
    output = graph.get_output(0)
    result = output.get_graph_output()
    if result is not None and output_path != "":
        cv2.imwrite(output_path, result)
    
    graph.deinit()
    nndeploy.base.time_profiler_print(graph.get_name())
   
if __name__ == "__main__":
    json_file, node_param_map, parallel_type, remove_in_out_node, input_path, output_path = parse_args()
    if remove_in_out_node:
        run_json_remove_in_out_node(json_file, node_param_map, parallel_type, input_path, output_path)
    else:
        run_json(json_file, node_param_map, parallel_type)
  