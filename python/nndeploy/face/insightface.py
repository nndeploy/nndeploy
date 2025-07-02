import nndeploy.base
import nndeploy.device
import nndeploy.dag

import os
import shutil
from typing import Any
import cv2
import json
import numpy as np
import insightface

class InsightFaceAnalysis(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceAnalysis")
        super().set_desc("InsightFace Analysis: get face analysis from image") 
        self.set_input_type(np.ndarray)
        self.set_output_type(list[Any])
        
        self.insightface_name_ = "buffalo_l"
        self.providers_ = ["CPUExecutionProvider"]
        self.is_one_face_ = True
        self.ctx_id = 0
        self.det_size_ = (640, 640)
        
    def init(self):
        self.analysis = insightface.app.FaceAnalysis(name=self.insightface_name_, providers=self.providers_)
        self.analysis.prepare(ctx_id=self.ctx_id, det_size=self.det_size_)
        return nndeploy.base.Status.ok()
        
    def run(self):
        input_numpy = self.get_input(0).get_numpy(self)
        faces = self.analysis.get(input_numpy)
        if len(faces) == 0:
            return nndeploy.base.Status(nndeploy.base.StatusCode.Error, "No face detected")
        if self.is_one_face_:
            face = min(faces, key=lambda x: x.bbox[0])
        self.get_output(0).set_any(face)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["insightface_name_"] = self.insightface_name_
        json_obj["providers_"] = self.providers_
        json_obj["is_one_face_"] = self.is_one_face_
        json_obj["ctx_id"] = self.ctx_id
        json_obj["det_size_"] = list(self.det_size_)
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.insightface_name_ = json_obj["insightface_name_"]
        self.providers_ = json_obj["providers_"]
        self.is_one_face_ = json_obj["is_one_face_"]
        self.ctx_id = json_obj["ctx_id"]
        self.det_size_ = tuple(json_obj["det_size_"])
        self.det_thresh_ = json_obj.get("det_thresh_", 0.5)
        return super().deserialize(target)
      
class InsightFaceAnalysisCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightFaceAnalysis(name, inputs, outputs)
        return self.node
      
insightface_node_creator = InsightFaceAnalysisCreator()
nndeploy.dag.register_node("nndeploy.face.InsightFaceAnalysis", insightface_node_creator)   


class InsightFaceSwapper(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceSwapper")
        super().set_desc("InsightFace Swapper: swap face from image")
        self.set_input_type(list[Any])
        self.set_input_type(list[Any])
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
        self.model_path_ = "inswapper_128_fp16.onnx"
        self.providers_ = ["CPUExecutionProvider"]
        
    def init(self):
        self.swapper = insightface.model_zoo.get_model(self.model_path_, providers=self.providers_)
        return nndeploy.base.Status.ok()
    
    def run(self):
        source_face = self.get_input(0).get_any(self)
        # print((source_face))
        target_face = self.get_input(1).get_any(self)
        # print((target_face))
        temp_frame = self.get_input(2).get_numpy(self)
        # print(type(temp_frame))
        # cv2.imwrite("temp_frame.jpg", temp_frame)
        self.swapped_frame = self.swapper.get(temp_frame, target_face, source_face, paste_back=True)
        # print(type(self.swapped_frame))
        self.get_output(0).set(self.swapped_frame)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["model_path_"] = self.model_path_
        json_obj["providers_"] = self.providers_
        return json.dumps(json_obj)
      
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.model_path_ = json_obj["model_path_"]
        self.providers_ = json_obj["providers_"]
        return super().deserialize(target)
      
class InsightFaceSwapperCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightFaceSwapper(name, inputs, outputs)
        return self.node
      
insightface_swapper_node_creator = InsightFaceSwapperCreator()
nndeploy.dag.register_node("nndeploy.face.InsightFaceSwapper", insightface_swapper_node_creator)   


class FaceSwapper(nndeploy.dag.Graph):
    def __init__(self, name: str):
        super().__init__(name)
        self.set_key("nndeploy.face.FaceSwapper")
        self.set_desc("FaceSwapper: swap face from image")
        self.set_input_type(np.ndarray)
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
        self.face_analysis_source = InsightFaceAnalysis("face_analysis_source")
        self.face_analysis_target = InsightFaceAnalysis("face_analysis_target")
        self.face_swapper = InsightFaceSwapper("face_swapper")
        
    def set_face_swapper_model_path(self, model_path: str):
        self.face_swapper.model_path_ = model_path
        
    def forward(self, inputs: [nndeploy.dag.Edge]):
        source_face = self.face_analysis_source([inputs[0]])
        target_face = self.face_analysis_target([inputs[1]])
        swapped_frame = self.face_swapper([source_face[0], target_face[0], inputs[1]])
        return swapped_frame
        
class FaceSwapperCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = FaceSwapper(name, inputs, outputs)
        return self.node
      
face_swapper_node_creator = FaceSwapperCreator()
nndeploy.dag.register_node("nndeploy.face.InsightFaceSwapper", insightface_swapper_node_creator)   
      
      
      
        
        