# executor.py

import time
import traceback
import logging
import json
from typing import Dict, Tuple, Any, List

from nndeploy.dag import GraphRunner

class GraphExecutor:
    """Encapsulate nndeploy load and run logic"""
    def __init__(self, cache_type=False, cache_size=None):
        self.runner = GraphRunner()

    def execute(self, graph_json: Dict, task_id: str) -> Tuple[Dict, float]:
        name = graph_json.get("name_")
        if isinstance(graph_json, (dict, list)):
            graph_json = json.dumps(graph_json, ensure_ascii=False)

        return self.runner.run(graph_json, name, task_id)
