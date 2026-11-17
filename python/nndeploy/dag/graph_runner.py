# graph_runner.py

from __future__ import annotations
import json
import time
import copy
import logging
import traceback
from typing import Dict, Any, Tuple, List
import argparse
import threading

import nndeploy.base
import nndeploy.device
from .graph import Graph
from .node import Node, add_global_import_lib, import_global_import_lib
from .edge import Edge
from .base import EdgeTypeInfo
from .base import io_type_to_name

class NnDeployGraphRuntimeError(RuntimeError):
    def __init__(self, status, msg):
        self.status = status
        self.msg = msg
        super().__init__(f"Graph run failed: {status}, {msg}")

class GraphRunner:
    def __init__(self):
        self.graph = None
        self.is_cancel = False
        self.args = None
        self._owner = None
        self._close_error = None

    def load(self, graph_json=None, *, graph=None, graph_factory=None, before_init=None):
        """Initialize a resident graph on its execution thread.

        The existing run() remains the one-shot editor path. Resident callers
        use load(), execute() and close() on one owning thread.
        """
        if self.graph is not None:
            raise RuntimeError("GraphRunner already owns a graph")
        if sum(value is not None for value in (graph_json, graph, graph_factory)) != 1:
            raise ValueError("Provide exactly one graph source")
        self._owner = threading.get_ident()
        self.is_cancel = False
        try:
            if graph_json is not None:
                data = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
                self.graph = Graph(data.get("name_", "server"))
                self._check_status(self.graph.deserialize(json.dumps(data, ensure_ascii=False)))
            else:
                candidate = graph_factory() if graph_factory is not None else graph
                if candidate.get_initialized():
                    raise ValueError("Server requires an uninitialized Graph with exclusive ownership")
                self.graph = candidate
            parallel = self.graph.get_parallel_type()
            if parallel not in (nndeploy.base.ParallelType.kParallelTypeNone, nndeploy.base.ParallelType.Sequential):
                raise ValueError("Resident execution currently requires a sequential Graph")
            if before_init is not None:
                before_init(self.graph)
            self._check_status(self.graph.init())
            if self.graph.get_loop_count() != 1:
                raise ValueError("Resident execution requires one finite invocation")
            return self.graph
        except BaseException:
            self.close()
            raise

    def _check_owner(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("Resident Graph must be used on its owning thread")

    def execute(self, bind, collect, *, steps=None, after_step=None):
        """Bind a request, run its finite steps, then detach its outputs.

        The caller bounds the step count after binding inputs; ordinary requests
        run once. Per-step checks run after synchronization, before collection.
        """
        self._check_owner()
        if self._close_error is not None:
            raise self._close_error
        if self.graph is None:
            raise RuntimeError("Graph is not loaded")
        try:
            with bind(self.graph):
                count = steps() if steps is not None else 1
                if type(count) is not int or count < 1:
                    raise ValueError('Resident Graph execution requires a finite positive step count')
                for _ in range(count):
                    self._check_status(self.graph.run())
                    if not self.graph.synchronize():
                        raise RuntimeError("Graph synchronization failed")
                    if after_step is not None:
                        after_step()
                return collect(self.graph)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._close_error is not None:
            raise self._close_error
        if self.graph is not None:
            self._check_owner()
            try:
                self._check_status(self.graph.deinit())
            except BaseException as exc:
                # A second cleanup attempt cannot prove that a failed native
                # deinit released its threads, devices or model file readers.
                self._close_error = exc
                raise
            self.graph = None

    def _check_status(self, status):
        if status != nndeploy.base.StatusCode.Ok:
            msg = status.get_desc()
            raise NnDeployGraphRuntimeError(status, msg)

    def _build_graph(self, graph_json_str: str, name: str):
        self.graph = Graph(name)
        if self.args is not None:
            for item in self.args.node_param:
                self.graph.set_node_value(item)
        status = self.graph.deserialize(graph_json_str)
        return self.graph, status
    
    def cancel_running(self):
        if self.graph is not None:
            flag = self.graph.interrupt()
            self.is_cancel = True
            if not flag:
                raise RuntimeError(f"interrupt failed")

    def get_run_status(self):
        if self.graph is None:
            return "{}"
        run_status_map = self.graph.get_nodes_run_status_recursive()
        # print(f"run_status_map: {run_status_map}")
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
        
    def get_loop_count(self):
        if self.graph is not None:
            return self.graph.get_loop_count()
        return 0
    
    def get_parallel_type(self):
        if self.graph is not None:
            return self.graph.get_parallel_type()
        return nndeploy.base.ParallelType.Pipeline

    def run(self, graph_json_str: str, name: str, task_id: str, args: GraphRunnerArgs = None,
            *, before_init=None, before_run=None) -> Tuple[Dict[str, Any], List[Any]]:
        self.is_cancel = False
        did_deinit = False
        self.args = args
        try:           
            nndeploy.base.time_profiler_reset()
            if self.is_cancel:
                raise RuntimeError(f"graph interrupted!")

            nndeploy.base.time_point_start("deserialize_" + name)
            self.graph, status = self._build_graph(graph_json_str, name)
            self._check_status(status)
            nndeploy.base.time_point_end("deserialize_" + name)

            self.graph.set_time_profile_flag(True)
            if args is not None and args.debug:
                self.graph.set_debug_flag(True)
            if args is not None and getattr(args, "parallel_type", ""):
                self.graph.set_parallel_type(
                    nndeploy.base.ParallelType.from_name(args.parallel_type))

            if self.is_cancel:
                raise RuntimeError(f"graph interrupted!")

            nndeploy.base.time_point_start("init_" + name)
            if before_init is not None:
                before_init(self.graph)
            status = self.graph.init()
            self._check_status(status)
            nndeploy.base.time_point_end("init_" + name)

            parallel_type = self.graph.get_parallel_type()
            results = {}

            if args is not None and args.dump:
                self.graph.dump()

            if self.is_cancel:
                raise RuntimeError(f"graph interrupted!")

            if before_run is not None:
                before_run()
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
                            # 判断对象是否可以拷贝，如果可以拷贝，采用拷贝的方式
                            try:
                                copy_result = copy.deepcopy(result)
                            except (TypeError, AttributeError, RecursionError) as e:
                                # 如果无法深拷贝，则直接使用原对象
                                copy_result = result
                            comsumers = output.get_consumers()
                            for consumer in comsumers:
                                consumer_name = consumer.get_name()
                                io_type = io_type_to_name[consumer.get_io_type()]
                                if consumer_name not in results:
                                    results[consumer_name] = {}
                                if io_type not in results[consumer_name]:
                                    results[consumer_name][io_type] = []
                                results[consumer_name][io_type].append(copy_result)
            if parallel_type == nndeploy.base.ParallelType.Pipeline:
                for i in range(count):
                    outputs = self.graph.get_all_output()
                    for output in outputs:
                        result = output.get_graph_output()
                        if result is not None:
                            # 判断对象是否可以拷贝，如果可以拷贝，采用拷贝的方式
                            try:
                                copy_result = copy.deepcopy(result)
                            except (TypeError, AttributeError, RecursionError) as e:
                                # 如果无法深拷贝，则直接使用原对象
                                copy_result = result
                            comsumers = output.get_consumers()
                            for consumer in comsumers:
                                consumer_name = consumer.get_name()
                                io_type = io_type_to_name[consumer.get_io_type()]
                                if consumer_name not in results:
                                    results[consumer_name] = {}
                                if io_type not in results[consumer_name]:
                                    results[consumer_name][io_type] = []
                                results[consumer_name][io_type].append(copy_result)
            flag = self.graph.synchronize()
            if not flag:
                raise RuntimeError(f"synchronize failed")
            nndeploy.base.time_point_end("sum_" + name)

            if self.is_cancel:
                raise RuntimeError(f"graph interrupted!")

            nodes_name = self.graph.get_nodes_name_recursive()
            
            # print(time_profiler_map)
            nndeploy.base.time_profiler_print(name)
            if count > 10:   
                nndeploy.base.time_profiler_print_remove_warmup(name, 10)

            # 另一个线程启动的函数
            # run_status_map = self.get_run_status()
            # print(run_status_map)

            if self.is_cancel:
                raise RuntimeError(f"graph interrupted!")

            nndeploy.base.time_point_start("deinit_" + name)
            status = self.graph.deinit()
            self._check_status(status)
            did_deinit = True

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

            # print(results)
            return time_profiler_map, results, status, status.get_desc()
        
        except NnDeployGraphRuntimeError as e:
            return {}, {}, e.status, e.msg

        finally:
            try:
                if self.graph is not None:
                    if not did_deinit:
                        try:
                            self.graph.deinit()
                        except Exception:
                            pass
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            finally:
                pass
