# graph_runner.py

from __future__ import annotations
import json
import time
import logging
import traceback
from typing import Dict, Any, Tuple, List
import argparse
import shutil
from pathlib import Path

import nndeploy.base
import nndeploy.device
from .node import add_global_import_lib, import_global_import_lib
from .graph_runner import GraphRunner
     
       
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--name", type=str, default="", required=False)
    parser.add_argument("--task_id", type=str, default="", required=False)
        
    # node_name:param_key:param_value
    parser.add_argument("--node_param", "-np", type=str, nargs='*', default=[], required=False)
    
    # kParallelTypeSequential
    parser.add_argument("--parallel_type", type=str, default="", required=False)
    
    # dump
    parser.add_argument("--dump", action="store_true", default=False, required=False)
    # debug
    parser.add_argument("--debug", action="store_true", default=False, required=False)
    
    # plugin
    parser.add_argument("--plugin", type=str, nargs='*', default=[], required=False)
    # resources
    parser.add_argument("--resources", type=str, default="", required=False)
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
        self.debug = False
        self.plugin = []
        self.resources = ""
        
    def parse_args(self):
        args = parse_args()
        self.json_file = args.json_file
        self.name = args.name
        self.task_id = args.task_id
        
        for item in args.node_param:
            self.node_param.append(item)
        
        for item in args.plugin:
            self.plugin.append(item)
        
        self.parallel_type = args.parallel_type
        self.dump = args.dump
        self.debug = args.debug
        self.resources = args.resources

# def main():
#     args = GraphRunnerArgs()
#     args.parse_args()

#     if args.plugin != []:
#         for plugin_path in args.plugin:
#             add_global_import_lib(plugin_path)
#         import_global_import_lib()

#     graph_json_str = ""
#     with open(args.json_file, "r") as f:
#         graph_json_str = f.read()
#     gr = GraphRunner()
#     time_profiler_map, results, _, _ = gr.run(graph_json_str, args.name, args.task_id, args)
#     # print(time_profiler_map)
#     # print(results)

def main():
    args = GraphRunnerArgs()
    args.parse_args()

    logger = logging.getLogger("run_json")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Step 1: copy resources directory
    if args.resources:
        src = Path(args.resources).resolve()
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
            return
        else:
            try:
                shutil.copytree(src, dst)
                logger.info(f"Resources copied successfully to {dst}")
            except Exception as e:
                logger.error(f"Failed to copy resources from {src} to {dst}: {e}")
                return

    # Step 2: load plugins
    if args.plugin:
        for plugin_path in args.plugin:
            add_global_import_lib(plugin_path)
        import_global_import_lib()
        logger.info(f"Loaded plugins: {args.plugin}")

    # Step 3: load JSON file and run
    try:
        with open(args.json_file, "r") as f:
            graph_json_str = f.read()
    except Exception as e:
        logger.error(f"Failed to read JSON file {args.json_file}: {e}")
        return

    gr = GraphRunner()
    try:
        time_profiler_map, results, _, _ = gr.run(graph_json_str, args.name, args.task_id, args)
        logger.info("GraphRunner execution completed successfully.")
    except Exception as e:
        logger.error(f"GraphRunner execution failed: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()