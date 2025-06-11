# executor.py

import time
import traceback
import logging
from typing import Dict, Tuple, Any, List

import nndeploy

class GraphExecutor:
    """Encapsulate nndeploy load and run logic"""
    def __init__(self, server: "NnDeployServer", cache_type=False, cache_size=None):
        self.server = server
    
    def execute(self, graph_json: Dict, task_id: str) -> Tuple[Dict, float]:
        try:
            graph = nndeploy.Graph.from_json(graph_json)
        except Exception as e:
            raise RuntimeError(f"解析 JSON 构图失败: {e}")
    
    def _progress_hook(val, total, preview=None):
        payload = {"value": val, "max": total, "task_id": task_id}
        self.server.send_sync("progress", payload)
    graph.set_progress_hook(_progress_hook)

    t0 = time.perf_counter()
    graph.run()
    t1 = time.perf_counter()

    outputs: Dict[str, Any] = graph.collect_outputs()

    return outputs, t1 - t0
        