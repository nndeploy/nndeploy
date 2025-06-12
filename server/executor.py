# executor.py

import time
import traceback
import logging
import json
from typing import Dict, Tuple, Any, List

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.detect
import nndeploy._nndeploy_internal as _C

class YoloDemo(nndeploy.dag.Graph):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("YoloDemo")
        self.set_output_type(nndeploy.detect.DetectResult)
        self.decodec = self.create_node("nndeploy::codec::OpenCvImageDecodeNode", "decodec")
        # self.yolo = self.create_node("nndeploy::detect::YoloPyGraph", "yolo")
        self.yolo = nndeploy.detect.YoloPyGraph("yolo")
        self.add_node(self.yolo)
        self.drawbox = self.create_node("nndeploy::detect::DrawBoxNode", "drawbox")
        self.encodec = self.create_node("nndeploy::codec::OpenCvImageEncodeNode", "encodec")

    def forward(self, inputs: [nndeploy.dag.Edge] = []):
        decodec_outputs = self.decodec(inputs)
        yolo_outputs = self.yolo(decodec_outputs)
        drawbox_outputs = self.drawbox([decodec_outputs[0], yolo_outputs[0]])
        self.encodec(drawbox_outputs)
        return yolo_outputs

    def make(self, decodec_desc, yolo_desc, drawbox_desc, encodec_desc):
        self.set_node_desc(self.decodec, decodec_desc)
        self.set_node_desc(self.yolo, yolo_desc)
        self.set_node_desc(self.drawbox, drawbox_desc)
        self.set_node_desc(self.encodec, encodec_desc)
        return nndeploy.base.StatusCode.Ok

    def get_yolo(self):
        return self.yolo

    def set_size(self, size):
        self.decodec.set_size(size)

    def set_input_path(self, path):
        self.decodec.set_path(path)

    def set_output_path(self, path):
        self.encodec.set_path(path)

class GraphExecutor:
    """Encapsulate nndeploy load and run logic"""
    def __init__(self, server: "NnDeployServer", cache_type=False, cache_size=None):
        self.server = server

    def execute(self, graph_json: Dict, task_id: str) -> Tuple[Dict, float]:
        if isinstance(graph_json, (dict, list)):
            graph_json = json.dumps(graph_json, ensure_ascii=False)
        try:
            # graph = nndeploy.Graph.from_json(graph_json)
            yolo_demo = YoloDemo("yolo_demo")
            yolo_demo.deserialize(graph_json)
        except Exception as e:
            raise RuntimeError(f"解析 JSON 构图失败: {e}")

        def _progress_hook(val, total, preview=None):
            payload = {"value": val, "max": total, "task_id": task_id}
            self.server.send_sync("progress", payload)
        # graph.set_progress_hook(_progress_hook)

        t0 = time.perf_counter()
        inputs: [nndeploy.dag.Edge] = []
        outputs = yolo_demo(inputs)
        t1 = time.perf_counter()

        # outputs: Dict[str, Any] = graph.collect_outputs()

        return outputs, t1 - t0
