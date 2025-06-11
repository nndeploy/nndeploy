
import nndeploy._nndeploy_internal as _C

# from nndeploy._nndeploy_internal import Node, NodeDesc, Graph

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.detect
import torch

import sys


# python3 nndeploy/test/detect/test_detect.py

# 创建的所有节点都最好都是成员变量self.xxx，不要在forward中创建节点
class YoloDemo(nndeploy.dag.Graph):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("YoloDemo")
        self.set_output_type(nndeploy.detect.DetectResult)
        self.decodec = self.create_node("nndeploy::codec::OpenCvImageDecodeNode", "decodec")
        self.yolo = self.create_node("nndeploy::detect::YoloPyGraph", "yolo")
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
        
    
def test_yolo():    
    yolo_demo = YoloDemo("yolo_demo")
    yolo = yolo_demo.get_yolo()
    yolo.default_param()
    yolo.set_inference_type(nndeploy.base.InferenceType.OnnxRuntime)
    yolo.set_version(11)
    yolo.set_infer_param(nndeploy.base.DeviceType("x86"), nndeploy.base.ModelType.Onnx, True, ["/home/always/github/public/nndeploy/build/yolo11s.sim.onnx"])
    yolo_demo.set_time_profile_flag(True)
    
    inputs: [nndeploy.dag.Edge] = []
    output = yolo_demo.trace(inputs)
    
    yolo_demo.set_input_path("/home/always/github/public/nndeploy/docs/image/demo/detect/sample.jpg")
    count = 1
    yolo_demo.set_size(count)
    yolo_demo.set_output_path("/home/always/github/public/nndeploy/build/yolo_python_demo.jpg")
    nndeploy.base.time_point_start("yolo_demo_python")
    for i in range(count):
        print(f"test_yolo {i}")
        output = yolo_demo(inputs)
        result = output[0].get_graph_output_param()
        for bbox in result.bboxs_:
            print(f"Class ID: {bbox.label_id_}, Confidence: {bbox.score_:.2f}, Bounding Box: {bbox.bbox_}")
    nndeploy.base.time_point_end("yolo_demo_python")
    
    yolo_demo.save_file("/home/always/github/public/nndeploy/build/yolo_demo.json")
        
    nndeploy.base.time_profiler_print("yolo_demo")
    
    
def test_yolo_from_json():
    yolo_demo = YoloDemo("yolo_demo")
    yolo_demo.load_file("/home/always/github/public/nndeploy/build/yolo_demo.json")
    yolo_demo.set_time_profile_flag(True)
    yolo_demo.set_input_path("/home/always/github/public/nndeploy/docs/image/demo/detect/sample.jpg")
    count = 1
    yolo_demo.set_size(count)
    yolo_demo.set_output_path("/home/always/github/public/nndeploy/build/yolo_jos_python_demo.jpg")
    nndeploy.base.time_point_start("test_yolo_from_json")
    inputs: [nndeploy.dag.Edge] = []
    for i in range(count):
        print(f"test_yolo_from_json {i}")
        output = yolo_demo(inputs)
        result = output[0].get_graph_output_param()
        for bbox in result.bboxs_:
            print(f"Class ID: {bbox.label_id_}, Confidence: {bbox.score_:.2f}, Bounding Box: {bbox.bbox_}")
    nndeploy.base.time_point_end("test_yolo_from_json")
    nndeploy.base.time_profiler_print("test_yolo_from_json")
    

    
if __name__ == "__main__":
    test_yolo()
    test_yolo_from_json()
    
        
        
        
        
