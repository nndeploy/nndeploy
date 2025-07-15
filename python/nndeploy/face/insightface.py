import nndeploy.base
import nndeploy.device
import nndeploy.dag

import os
import shutil
from typing import Any
import cv2
import json
from tqdm import tqdm
import numpy as np
import insightface

from .deep_live_cam import create_face_mask, create_lower_mouth_mask, apply_mouth_area, draw_mouth_mask_visualization
from .deep_live_cam import find_cluster_centroids, find_closest_centroid

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
        input_numpy = self.get_input(0).get(self)
        faces = self.analysis.get(input_numpy)
        print(type(faces))
        if len(faces) == 0:
            print("No face detected")
            face = faces  # 返回空列表，保持与faces一致的类型
        else:
            if self.is_one_face_:
                selected_face = min(faces, key=lambda x: x.bbox[0])
                face = [selected_face]  # 返回包含单个face的列表，保持与faces一致的类型
                print(type(face))
            else:
                face = faces  # 返回所有faces，保持原有类型
        self.get_output(0).set(face)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["insightface_name_"] = self.insightface_name_
        json_obj["providers_"] = self.providers_
        json_obj["is_one_face_"] = self.is_one_face_
        json_obj["ctx_id"] = self.ctx_id
        # json_obj["det_size_"] = list(self.det_size_)
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.insightface_name_ = json_obj["insightface_name_"]
        self.providers_ = json_obj["providers_"]
        self.is_one_face_ = json_obj["is_one_face_"]
        self.ctx_id = json_obj["ctx_id"]
        # self.det_size_ = tuple(json_obj["det_size_"])
        # self.det_thresh_ = json_obj.get("det_thresh_", 0.5)
        return super().deserialize(target)
      
class InsightFaceAnalysisCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightFaceAnalysis(name, inputs, outputs)
        return self.node
      
insightface_node_creator = InsightFaceAnalysisCreator()
nndeploy.dag.register_node("nndeploy.face.InsightFaceAnalysis", insightface_node_creator)   


class InsightImageFaceId(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceId")
        super().set_desc("InsightFace Id: get face id from image")
        self.set_input_type(np.ndarray)
        self.set_input_type(list[Any])
        self.set_output_type(list[Any])
        
    def run(self):
        input_numpy = self.get_input(0).get(self)
        faces = self.get_input(1).get(self)
        source_target_map = []
        i = 0
        for face in faces:
            x_min, y_min, x_max, y_max = face['bbox']
            source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : input_numpy[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
        self.get_output(0).set(source_target_map)
        return nndeploy.base.Status.ok()
    

class InsightImageFaceIdCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightImageFaceId(name, inputs, outputs)
        return self.node
     
      
insightimage_face_id_node_creator = InsightImageFaceIdCreator()
nndeploy.dag.register_node("nndeploy.face.InsightImageFaceId", insightimage_face_id_node_creator)   


class InsightVideoFaceId(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceId")
        super().set_desc("InsightFace Id: get face id from image")
        super().set_node_type(nndeploy.dag.NodeType.Input)
        
        self.video_path_ = "video.mp4"
        
        self.set_output_type(list[Any])
        import nndeploy.codec as codec
        self.codec_output = nndeploy.dag.Edge("codec_output")
        self.analysis_output = nndeploy.dag.Edge("analysis_output")
        self.video_codec = codec.OpenCvVedioDecode("video_codec", [], [self.codec_output])
        self.face_analysis = InsightFaceAnalysis("face_analysis", [self.codec_output], [self.analysis_output])
        self.face_analysis.is_one_face_ = False
        self.graph = nndeploy.dag.Graph("graph", [], [self.video_codec, self.face_analysis])
        self.graph.add_node(self.video_codec)
        self.graph.add_node(self.face_analysis)
      
    def init(self):
        status = self.graph.init()
        if status != nndeploy.base.Status.ok():
            print("Failed to init graph")
            return status
        return nndeploy.base.Status.ok()
      
    def run(self):
        size = self.video_codec.get_size()
        source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
        for i in range(size):
            self.graph.run()
            frame = self.video_codec.get_graph_output()
            faces = self.face_analysis.get_graph_output()
            if len(faces) == 0:
                print("No face detected")
                continue
            
            for face in faces:
                face_embeddings.append(face.normed_embedding)
            
            frame_face_embeddings.append({'frame': i, 'faces': faces, 'location': i})
            i += 1
        centroids = find_cluster_centroids(face_embeddings)
        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index       
        for i in range(len(centroids)):
            source_target_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

            source_target_map[i]['target_faces_in_frame'] = temp

        for map in source_target_map:
            best_face = None
            best_frame = None
            for frame in map['target_faces_in_frame']:
                if len(frame['faces']) > 0:
                    best_face = frame['faces'][0]
                    best_frame = frame
                    break

            for frame in map['target_faces_in_frame']:
                for face in frame['faces']:
                    if face['det_score'] > best_face['det_score']:
                        best_face = face
                        best_frame = frame

            x_min, y_min, x_max, y_max = best_face['bbox']

            # 从视频中得到第i帧图片
            cap = cv2.VideoCapture(self.video_path_)
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame['location'])
            ret, target_frame = cap.read()
            cap.release()
            map['target'] = {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : best_face
                            }
            
        self.get_output(0).set(source_target_map)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["video_path_"] = self.video_path_
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.video_path_ = json_obj["video_path_"]
        self.video_codec.set_path(self.video_path_)
        return super().deserialize(target)
    
class InsightVideoFaceIdCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightImageFaceId(name, inputs, outputs)
        return self.node
      
insightvideo_face_id_node_creator = InsightVideoFaceIdCreator()
nndeploy.dag.register_node("nndeploy.face.InsightVideoFaceId", insightvideo_face_id_node_creator) 


class InsightFaceSwapper(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceSwapper")
        super().set_desc("InsightFace Swapper: swap face from image")
        self.set_input_type(list[Any])
        self.set_input_type(list[Any])
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
        self.mouth_mask_ = False
        self.show_mouth_mask_box_ = False
        self.mask_down_size_ = 0.5
        self.mask_feather_ratio_ = 8
        self.mask_size_ = 1
        self.map_ = False
        self.model_path_ = "inswapper_128_fp16.onnx"
        self.providers_ = ["CPUExecutionProvider"]
        
    def init(self):
        self.swapper = insightface.model_zoo.get_model(self.model_path_, providers=self.providers_)
        return nndeploy.base.Status.ok()
    
    def run(self):
        source_face = self.get_input(0).get(self)
        target_face = self.get_input(1).get(self)
        temp_frame = self.get_input(2).get(self)
        for i, single_face in enumerate(target_face):
            if i == 0:
                swapped_frame = self.swapper.get(temp_frame, single_face, source_face[0], paste_back=True)
            else:
                swapped_frame = self.swapper.get(swapped_frame, single_face, source_face[0], paste_back=True)
            if self.mouth_mask_:
                face_mask = create_face_mask(single_face, temp_frame)

                 # Create the mouth mask
                mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
                    create_lower_mouth_mask(single_face, temp_frame, self.mask_down_size_, self.mask_size_)
                )

                # Apply the mouth area
                swapped_frame = apply_mouth_area(
                    swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon, self.mask_feather_ratio_
                )

                if self.show_mouth_mask_box_:
                    mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                    swapped_frame = draw_mouth_mask_visualization(
                        swapped_frame, single_face, mouth_mask_data, self.mask_feather_ratio_
                    )
        if len(target_face) == 0:
            swapped_frame = temp_frame
        self.get_output(0).set(swapped_frame)
        print(type(swapped_frame))
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["mouth_mask_"] = self.mouth_mask_
        json_obj["show_mouth_mask_box_"] = self.show_mouth_mask_box_
        json_obj["model_path_"] = self.model_path_
        json_obj["providers_"] = self.providers_
        return json.dumps(json_obj)
      
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.mouth_mask_ = json_obj["mouth_mask_"]
        self.show_mouth_mask_box_ = json_obj["show_mouth_mask_box_"]
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
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
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
nndeploy.dag.register_node("nndeploy.face.FaceSwapper", face_swapper_node_creator)   
      
      
      
        
        