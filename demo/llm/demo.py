
import json
import time
import logging
import traceback
from typing import Dict, Any, Tuple, List
import argparse
from pathlib import Path
import shutil

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.tokenizer

def copy_resources_to_current_directory(resources):
    logger = logging.getLogger("copy_resources_to_current_directory")
    if resources == "":
        return
    if resources:
        src = Path(resources).resolve()
        dst = Path.cwd() / src.name
        logger.info(f"Preparing to copy resources from {src} -> {dst}")

        if not src.exists() or not src.is_dir():
            logger.error(f"Resource directory not found or not a directory: {src}")
            return

        if dst.exists():
            logger.warning(
                f"Target directory already exists: {dst}. "
                "Skipping copy. Please remove or rename it manually if you want to overwrite."
            )
        else:
            try:
                shutil.copytree(src, dst)
                logger.info(f"Resources copied successfully to {dst}")
            except Exception as e:
                logger.error(f"Failed to copy resources from {src} to {dst}: {e}")
                return

def parse_args():
    parser = argparse.ArgumentParser()
    # json_file
    parser.add_argument("--json_file", type=str, required=True)
    # resources
    parser.add_argument("--resources", type=str, default="", required=False)
    # node_param: node_name:param_key:param_value
    parser.add_argument("--node_param", "-np", type=str, nargs='*', default=[], required=False)
    # parallel_type: kParallelTypeSequential
    parser.add_argument("--parallel_type", type=str, default="Sequential", required=False)
    # remove_in_out_node: True
    parser.add_argument("--remove_in_out_node", action="store_true", default=False, required=False)
    args = parser.parse_args()
    json_file = args.json_file
    node_param_map = {
      args.node_param[i].split(':')[0]: {args.node_param[i].split(':')[1]: args.node_param[i].split(':')[2]} for i in range(len(args.node_param))
    }
    parallel_type = nndeploy.base.name_to_parallel_type[args.parallel_type]
    remove_in_out_node = args.remove_in_out_node
    return json_file, node_param_map, parallel_type, remove_in_out_node

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
  
  
def run_json_remove_in_out_node(json_file, node_param_map, parallel_type):
    graph = nndeploy.dag.Graph("")
    graph.set_node_value(node_param_map)
    graph.set_time_profile_flag(True)
    graph.set_parallel_type(parallel_type)
    graph.remove_in_out_node()
    graph.load_file(json_file)
    graph.init()
    
    input = graph.get_input(0)    
    text = nndeploy.tokenizer.TokenizerText()
    text.texts_ = [ "<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n" ]
    input.set(text)
    status = graph.run()
    output = graph.get_output(0)
    result = output.get_graph_output()
    if result is not None:
        pass
    
    graph.deinit()
    nndeploy.base.time_profiler_print(graph.get_name())
   
if __name__ == "__main__":
    json_file, node_param_map, parallel_type, remove_in_out_node = parse_args()
    if remove_in_out_node:
        run_json_remove_in_out_node(json_file, node_param_map, parallel_type)
    else:
        run_json(json_file, node_param_map, parallel_type)
  