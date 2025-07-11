# graph_runner.py

from __future__ import annotations
import json
import time
import copy
import logging
import traceback
from typing import Dict, Any, Tuple, List
import argparse

import nndeploy.base
import nndeploy.device
from .graph import Graph
from .node import Node
from .edge import Edge
from .base import EdgeTypeInfo

class GraphRunner:
    def _build_graph(self, graph_json_str: str, name: str):
        graph = Graph(name)
        status = graph.deserialize(graph_json_str)
        if status != nndeploy.base.StatusCode.Ok:
            raise RuntimeError(f"deserialize failed: {status}")
        return graph
    
    def run(self, graph_json_str: str, name: str, task_id: str) -> Tuple[Dict[str, Any], List[Any]]:
        nndeploy.base.time_profiler_reset()
        
        nndeploy.base.time_point_start("deserialize_" + name)
        graph = self._build_graph(graph_json_str, name)
        nndeploy.base.time_point_end("deserialize_" + name)

        graph.set_time_profile_flag(True)
        graph.set_debug_flag(True)
        # graph.set_parallel_type(nndeploy.base.ParallelType.Task)
        # graph.set_parallel_type(nndeploy.base.ParallelType.Pipeline)
        
        nndeploy.base.time_point_start("init_" + name)
        status = graph.init()
        if status != nndeploy.base.StatusCode.Ok:
            raise RuntimeError(f"init failed: {status}")
        nndeploy.base.time_point_end("init_" + name)
        
        parallel_type = graph.get_parallel_type()
        results = []
        
        is_dump = True
        if is_dump:
            graph.dump()
        
        nndeploy.base.time_point_start("sum_" + name)
        count = graph.get_loop_count()
        for i in range(count):
            t0_0 = time.perf_counter()
            status = graph.run()
            if status != nndeploy.base.StatusCode.Ok:
                raise RuntimeError(f"run failed: {status}")
            t1_0 = time.perf_counter()
            print(f"run {i} times, time: {t1_0 - t0_0}")   
            if parallel_type != nndeploy.base.ParallelType.Pipeline:
                outputs = graph.get_all_output()
                for output in outputs:
                    result = output.get_graph_output()
                    if result is not None:
                        copy_result = copy.deepcopy(result)
                        results.append(copy_result)
        if parallel_type == nndeploy.base.ParallelType.Pipeline:
            for i in range(count):
                outputs = graph.get_all_output()
                for output in outputs:
                    result = output.get_graph_output()
                    if result is not None:
                        copy_result = copy.deepcopy(result)
                        results.append(copy_result)
        flag = graph.synchronize()
        if not flag:
            raise RuntimeError(f"synchronize failed")  
        nndeploy.base.time_point_end("sum_" + name)
        
        graph.deinit()
        
        nodes_name = graph.get_nodes_name_recursive()
        time_profiler_map = {}
        for node_name in nodes_name:
            time_profiler_map[node_name] = nndeploy.base.time_profiler_get_cost_time(node_name + " run()") / 1000.0
        time_profiler_map["sum_" + name] = nndeploy.base.time_profiler_get_cost_time("sum_" + name) / 1000.0
        time_profiler_map["init_" + name] = nndeploy.base.time_profiler_get_cost_time("init_" + name) / 1000.0
        time_profiler_map["deserialize_" + name] = nndeploy.base.time_profiler_get_cost_time("deserialize_" + name) / 1000.0
        
        print(time_profiler_map)
        nndeploy.base.time_profiler_print(name)
        
        # 另一个线程启动的函数
        run_status_map = graph.get_nodes_run_status_recursive()
        for node_name, run_status in run_status_map.items():
            print(f"{node_name}: {run_status.get_status()}, {run_status}")
            
        # graph.deinit()
        
        return time_profiler_map, results
        
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
    time_profiler_map, results = gr.run(graph_json_str, args.name, "test_graph_runner")
    print(time_profiler_map)
    print(results)

if __name__ == "__main__":
    main()


