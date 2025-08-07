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
from .node import Node, add_global_import_lib, import_global_import_lib, import_global_import_lib
from .edge import Edge
from .base import EdgeTypeInfo

class NnDeployGraphRuntimeError(RuntimeError):
    def __init__(self, status, msg):
        self.status = status
        self.msg = msg
        super().__init__(f"Graph run failed: {status}, {msg}")

class GraphRunner:
    def __init__(self):
        self.graph = None

    def _check_status(self, status):
        if status != nndeploy.base.StatusCode.Ok:
            msg = status.get_desc()
            raise NnDeployGraphRuntimeError(status, msg)

    def _build_graph(self, graph_json_str: str, name: str):
        self.graph = Graph(name)
        status = self.graph.deserialize(graph_json_str)
        return self.graph, status
    
    def get_run_status(self):
        if self.graph is None:
            return "{}"
        run_status_map = self.graph.get_nodes_run_status_recursive()
        json_obj = {}
        for node_name, run_status in run_status_map.items():
            status = run_status.get_status()
            if status == "INITING":
                json_obj[node_name] = {"time": -1.0, "status": status}
            elif status == "INITED":
                json_obj[node_name] = {"time": run_status.init_time, "status": status}
            else:
                json_obj[node_name] = {"time": run_status.average_time, "status": status}
        return json_obj

    def release(self):
        if self.graph is not None:
            self.graph = None
        import gc; gc.collect()

    def run(self, graph_json_str: str, name: str, task_id: str, args: GraphRunnerArgs = None) -> Tuple[Dict[str, Any], List[Any]]:
        try:
            nndeploy.base.time_profiler_reset()

            nndeploy.base.time_point_start("deserialize_" + name)
            self.graph, status = self._build_graph(graph_json_str, name)
            self._check_status(status)
            nndeploy.base.time_point_end("deserialize_" + name)

            self.graph.set_time_profile_flag(True)
            self.graph.set_debug_flag(False)
            # self.graph.set_parallel_type(nndeploy.base.ParallelType.Task)
            # self.graph.set_parallel_type(nndeploy.base.ParallelType.Pipeline)
            
            if args is not None:
                if args.parallel_type != "":
                    parallel_type = nndeploy.base.name_to_parallel_type(args.parallel_type)
                    self.graph.set_parallel_type(parallel_type)
                if args.input_path != {}:
                    i = 0
                    for key, value in args.input_path.items():
                        node = None
                        if key.isdigit():
                            node = self.graph.get_input_node(i)
                        else:
                            node = self.graph.get_node(key)
                        if node is not None:
                            if hasattr(node, 'set_path'):
                                node.set_path(value)
                            else:
                                print(f"Warning: node {node.getName() if hasattr(node, 'getName') else 'unknown'} not support set_path method")
                        i += 1
                if args.output_path != {}:
                    i = 0
                    for key, value in args.output_path.items():
                        node = None
                        if key.isdigit():
                            node = self.graph.get_output_node(i)
                        else:
                            node = self.graph.get_node(key)
                        if node is not None:
                            if hasattr(node, 'set_path'):
                                node.set_path(value)
                            else:
                                print(f"Warning: node {node.getName() if hasattr(node, 'getName') else 'unknown'} not support set_path method")
                        i += 1
                if args.node_param != {}:
                    for node_name, param_map in args.node_param.items():
                        for param_key, param_value in param_map.items():
                            node = self.graph.get_node(node_name)
                            if node is not None:
                                node.set_param(param_key, param_value)

            nndeploy.base.time_point_start("init_" + name)
            status = self.graph.init()
            self._check_status(status)
            nndeploy.base.time_point_end("init_" + name)

            parallel_type = self.graph.get_parallel_type()
            results = []

            if args is not None and args.dump:
                self.graph.dump()

            nndeploy.base.time_point_start("sum_" + name)
            count = self.graph.get_loop_count()
            for i in range(count):
                t0_0 = time.perf_counter()
                status = self.graph.run()
                self._check_status(status)
                t1_0 = time.perf_counter()
                print(f"run {i} times, time: {t1_0 - t0_0}")
                if parallel_type != nndeploy.base.ParallelType.Pipeline:
                    outputs = self.graph.get_all_output()
                    for output in outputs:
                        result = output.get_graph_output()
                        if result is not None:
                            copy_result = copy.deepcopy(result)
                            results.append(copy_result)
            if parallel_type == nndeploy.base.ParallelType.Pipeline:
                for i in range(count):
                    outputs = self.graph.get_all_output()
                    for output in outputs:
                        result = output.get_graph_output()
                        if result is not None:
                            copy_result = copy.deepcopy(result)
                            results.append(copy_result)
            flag = self.graph.synchronize()
            if not flag:
                raise RuntimeError(f"synchronize failed")
            nndeploy.base.time_point_end("sum_" + name)
            
            nodes_name = self.graph.get_nodes_name_recursive()
            
            # print(time_profiler_map)
            nndeploy.base.time_profiler_print(name)

            # 另一个线程启动的函数
            # run_status_map = self.get_run_status()
            # print(run_status_map)

            nndeploy.base.time_point_start("deinit_" + name)
            status = self.graph.deinit()
            self._check_status(status)

            is_release_cuda_cache = True
            if is_release_cuda_cache:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            nndeploy.base.time_point_end("deinit_" + name)

            # run_status_map = self.get_run_status()
            # print(run_status_map)

            # time_profiler_map = {}
            # for node_name in nodes_name:
            #     time_profiler_map[node_name] = nndeploy.base.time_profiler_get_cost_time(node_name + " run()")
            # time_profiler_map["sum_" + name] = nndeploy.base.time_profiler_get_cost_time("sum_" + name)
            # time_profiler_map["init_" + name] = nndeploy.base.time_profiler_get_cost_time("init_" + name)
            # time_profiler_map["deserialize_" + name] = nndeploy.base.time_profiler_get_cost_time("deserialize_" + name)

            time_profiler_map = {}
            time_profiler_map["init_time"] = nndeploy.base.time_profiler_get_cost_time("init_" + name)
            time_profiler_map["run_time"] = nndeploy.base.time_profiler_get_cost_time("sum_" + name)

            return time_profiler_map, results, status, status.get_desc()
        except NnDeployGraphRuntimeError as e:
            return {}, {}, e.status, e.msg
        
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
    return parser.parse_args()


class GraphRunnerArgs:
    def __init__(self):
        self.json_file = ""
        self.name = ""
        self.task_id = ""
        self.input_path = {}
        self.output_path = {}
        self.node_param = {}
        self.parallel_type = ""
        self.dump = False
        
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
                self.input_path[i] = item
            i += 1
        
        # 解析输出路径参数
        i = 0
        for item in args.output_path:
            if ":" in item:
                key, value = item.split(":", 1)
                self.output_path[key] = value
            else:
                self.output_path[i] = item
            i += 1
        
        i = 0
        for item in args.node_param:
            node_name, param_key, param_value = item.split(":")
            self.node_param[node_name][param_key] = param_value
            i += 1
        
        self.parallel_type = args.parallel_type
        self.dump = args.dump


def main():
    args = GraphRunnerArgs()
    args.parse_args()
    
    graph_json_str = ""
    with open(args.json_file, "r") as f:
        graph_json_str = f.read()
    gr = GraphRunner()
    time_profiler_map, results, _, _ = gr.run(graph_json_str, args.name, args.task_id, args)
    print(time_profiler_map)
    print(results)

if __name__ == "__main__":
    main()


