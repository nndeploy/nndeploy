# # graph_runner.py

# from __future__ import annotations
# import json
# import time
# import logging
# import traceback
# from typing import Dict, Any, Tuple, List

# import nndeploy.base
# import nndeploy.device
# import nndeploy.dag
# import nndeploy.detect

# class GraphRunner:
#     def _build_graph(self, graph_json_str: str, name: str):
#         graph = nndeploy.dag.Graph(name)
#         status = graph.deserialize(graph_json_str)
#         if status != nndeploy.base.StatusCode.Ok:
#             raise RuntimeError(f"deserialize failed: {status}")
#         return graph
    
#     def run(self, graph_json_str: str, name: str, task_id: str) -> Tuple[Dict[str, Any], float]:
#         graph = self._build_graph(graph_json_str, name)

#         graph.set_time_profile_flag(True)
        
#         status = graph.init()
#         if status != nndeploy.base.StatusCode.Ok:
#             raise RuntimeError(f"init failed: {status}")
        
#         t0 = time.perf_counter()
        
#         nndeploy.base.time_point_start(name)
#         count = graph.get_loop_count()
#         for i in range(count):
#             t0_0 = time.perf_counter()
#             status = graph.run()
#             if status != nndeploy.base.StatusCode.Ok:
#                 raise RuntimeError(f"run failed: {status}")
#             t1_0 = time.perf_counter()
#             print(f"run {i} times, time: {t1_0 - t0_0}")     
#         flag = graph.synchronize()
#         if not flag:
#             raise RuntimeError(f"synchronize failed")
#         nndeploy.base.time_point_end(name)
#         nndeploy.base.time_profiler_print(name)
        
#         t1 = time.perf_counter()
        
#         return t1 - t0
