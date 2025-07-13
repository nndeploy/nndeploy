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
from nndeploy.dag import GraphRunner
     

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--name", type=str, default="", required=False)
    return parser.parse_args()

def main():
    is_while = True
    args = parse_args()
    graph_json_str = ""
    with open(args.json_file, "r") as f:
        graph_json_str = f.read()
    gr = GraphRunner()
    time_profiler_map, results = gr.run(graph_json_str, args.name, "test_graph_runner")
    del gr
    
    import torch
    torch.cuda.empty_cache()
        
    while is_while:
        time.sleep(10)
        print(time_profiler_map)
        print(results)

if __name__ == "__main__":
    main()

# python nndeploy/test/dag/test_graph_runner.py --json_file /home/always/github/public/nndeploy/server/resources/workflow/face_swap_2025_07_04_v4.json