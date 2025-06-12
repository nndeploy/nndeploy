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

class GraphRunner:
    def __init__(self, server: "NnDeployServer"):
        self.server = server
    
    def _build_graph(self, graph_json_str: str) -> YoloDemo:
        graph = YoloDemo("yolo_demo")
        status = graph.deserialize(graph_json_str)
        if status != nndeploy.base.StatusCode.Ok:
            raise RuntimeError(f"deserialize failed: {status}")
        return graph
    
    def run(self, graph_json_str: str, task_id: str) -> Tuple[Dict[str, Any], float]:
        graph = self._build_graph(graph_json_str)

        t0 = time.perf_counter()
        outputs = graph([])
        t1 = time.perf_counter()

        return outputs, t1 - t0
