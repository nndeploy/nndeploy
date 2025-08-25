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
from .node import Node, add_global_import_lib, import_global_import_lib
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
            
            # Update graph_json_str
            if args is not None and args.node_param != []:
                graph_json_obj = json.loads(graph_json_str)
                for node_param in args.node_param:
                    params = node_param.split(":")
                    node_name = params[0]
                    node_keys = params[1:-1]
                    node_value = params[-1]
                    # 递归查找节点，因为node_repository_可能存在嵌套
                    def find_node_in_repository(repository, target_name):
                        for node in repository:
                            if node.get("name_") == target_name:
                                return node
                            # 递归查找嵌套的node_repository_
                            if "node_repository_" in node and node["node_repository_"]:
                                found = find_node_in_repository(node["node_repository_"], target_name)
                                if found:
                                    return found
                        return None
                    
                    node_json_obj = find_node_in_repository(graph_json_obj["node_repository_"], node_name)
                    if node_json_obj is None:
                        print(f"Warning: not found {node_name}")
                        continue
                
                # 处理多级嵌套的node_keys
                current_obj = node_json_obj
                for key in node_keys:
                    if key in current_obj:
                        current_obj = current_obj[key]
                    else:
                        print(f"Warning: key {key} not found in node {node_name}")
                        current_obj = None
                        break
                
                # 如果所有keys都找到了，设置最终的值
                if current_obj is not None:
                    # 获取父对象和最后一个key
                    parent_obj = node_json_obj
                    for key in node_keys[:-1]:
                        parent_obj = parent_obj[key]
                    parent_obj[node_keys[-1]] = node_value
                else:
                    print(f"Warning: no keys provided for node {node_name}")
                    
                
                graph_json_str = json.dumps(graph_json_obj)

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
                            # copy_result = copy.deepcopy(result)
                            copy_result = result
                            results.append(copy_result)
            if parallel_type == nndeploy.base.ParallelType.Pipeline:
                for i in range(count):
                    outputs = self.graph.get_all_output()
                    for output in outputs:
                        result = output.get_graph_output()
                        if result is not None:
                            # copy_result = copy.deepcopy(result)
                            copy_result = result
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