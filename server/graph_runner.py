# graph_runner.py

from __future__ import annotations
import json
import time
import logging
import traceback
from typing import Dict, Any, Tuple, List

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.detect

class GraphRunner:
    def _build_graph(self, graph_json_str: str, name: str):
        graph = nndeploy.dag.Graph(name)
        status = graph.deserialize(graph_json_str)
        if status != nndeploy.base.StatusCode.Ok:
            raise RuntimeError(f"deserialize failed: {status}")
        return graph
    
    def run(self, graph_json_str: str, name: str, task_id: str) -> Tuple[Dict[str, Any], float]:
        graph = self._build_graph(graph_json_str, name)

        t0 = time.perf_counter()
        graph.init()
        graph.run()
        t1 = time.perf_counter()

        return t1 - t0
