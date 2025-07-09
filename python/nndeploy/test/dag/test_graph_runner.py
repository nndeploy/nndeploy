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

# class GraphRunner:
#     def __init__(self):
#         pass
    
#     def _build_graph(self, graph_json_str: str, name: str):
#         graph = nndeploy.dag.Graph(name)
#         status = graph.deserialize(graph_json_str)
#         if status != nndeploy.base.StatusCode.Ok:
#             raise RuntimeError(f"deserialize failed: {status}")
#         return graph
    
#     def run(self, graph_json_str: str, name: str, task_id: str) -> Tuple[Dict[str, Any], float]:
#         graph = self._build_graph(graph_json_str, name)

#         t0 = time.perf_counter()
#         graph.init()
        
#         count_map = graph.get_loop_count_map()
#         print(f"count_map: {count_map}")
        
#         count = 0
#         for k, v in count_map.items():
#             count = max(count, v)
#         print(f"count: {count}")

#         for i in range(count):
#             t0_0 = time.perf_counter()
#             graph.run()
#             t1_0 = time.perf_counter()
#             print(f"run {i} times, time: {t1_0 - t0_0}")
            
#         t1 = time.perf_counter()
        
#         print(f"time: {t1 - t0}")

#         # outputs = graph.get_all_output()

#         return t1 - t0
      

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--name", type=str, default="", required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    graph_json_str = ""
    with open(args.json_file, "r") as f:
        graph_json_str = f.read()
    gr = GraphRunner()
    gr.run(graph_json_str, args.name, "test_graph_runner")

if __name__ == "__main__":
    main()

# python nndeploy/test/dag/test_graph_runner.py --json_file /home/always/github/public/nndeploy/server/resources/workflow/face_swap_2025_07_04_v4.json