
import nndeploy._nndeploy_internal as _C

# from nndeploy._nndeploy_internal import Node, NodeDesc, Graph

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import torch


# python3 nndeploy/test/detect/test_detect.py
      
      
class DetectGraph(nndeploy.dag.Graph):
    def __init__(self, name, outputs: _C.dag.Edge):
        super().__init__(name, [], [outputs])
        self.model_inputs = ["images"]
        self.model_outputs = ["output0"]
        self.inference_type = nndeploy.base.InferenceType.OnnxRuntime
        self.version = 11
        self.device_type = nndeploy.base.DeviceType.Arm
        self.model_type = nndeploy.base.ModelType.Onnx
        self.is_path = True
        self.model_value = "/home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx"
        
        self.input_edge = self.create_edge("input_edge")
        self.yolo_graph = _C.detect.YoloGraph("yolo_graph", [self.input_edge], [outputs])
        pre_desc = _C.dag.NodeDesc("preprocess", ["detect_in"], self.model_inputs)
        infer_desc = _C.dag.NodeDesc("infer", self.model_inputs, self.model_outputs) 
        post_desc = _C.dag.NodeDesc("postprocess", self.model_outputs, ["detect_out"])
        self.yolo_graph.make(pre_desc, infer_desc, self.inference_type, post_desc)
        self.yolo_graph.set_infer_param(self.device_type, self.model_type, self.is_path, self.model_value)
        self.yolo_graph.set_version(self.version)
        self.add_node_shared_ptr(self.yolo_graph)
        
        self.decode_node = _C.codec.create_decode_node_shared_ptr(nndeploy.base.kCodecTypeOpenCV, nndeploy.base.kCodecFlagImage, "decode_node", self.input_edge)
        self.add_node_shared_ptr(self.decode_node)
        
        self.draw_output = self.create_edge("draw_output")
        self.draw_box_node = _C.detect.DrawBoxNode("draw_box_node", [self.input_edge, outputs], [self.draw_output])
        self.add_node_shared_ptr(self.draw_box_node)
        
        self.encode_node = _C.codec.create_encode_node_shared_ptr(nndeploy.base.kCodecTypeOpenCV, nndeploy.base.kCodecFlagImage, "encode_node", self.draw_output)
        self.add_node_shared_ptr(self.encode_node)
    
    def set_input_path(self, path):
        self.decode_node.set_path(path)
        
    def set_output_path(self, path):
        self.encode_node.set_path(path)
        
    def set_inference_param(self, device_type, model_type, is_path, model_value):
        self.yolo_graph.set_infer_param(device_type, model_type, is_path, model_value)
        
    def set_version(self, version):
        self.yolo_graph.set_version(version)
        
    def set_src_pixel_type(self, pixel_type):
        self.yolo_graph.set_src_pixel_type(pixel_type)
        
    
def test_detect():
    # 创建detect_graph
    outputs = _C.dag.Edge("outputs")
    detect_graph = DetectGraph("detect_graph", outputs)
    
    detect_graph.init()
    detect_graph.dump()
    
    # 设置输入路径
    detect_graph.set_input_path("/home/ascenduserdg01/github/nndeploy/docs/image/demo/detect/sample.jpg")
    # 设置输出路径
    detect_graph.set_output_path("/home/ascenduserdg01/github/nndeploy/build/py_sample_output.jpg")
    detect_graph.run()    
    
    
if __name__ == "__main__":
    test_detect()
    
    
        
        
        
        
