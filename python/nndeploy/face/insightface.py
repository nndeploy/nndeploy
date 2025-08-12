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
        self.set_output_type(list[insightface.app.common.Face])
        
        self.insightface_name_ = "buffalo_l"
        self.providers_ = ["CPUExecutionProvider"]
        self.is_one_face_ = True
        self.ctx_id = 0
        self.det_size_ = (640, 640)
        self.det_thresh_ = 0.5
        
    def init(self):
        self.analysis = insightface.app.FaceAnalysis(name=self.insightface_name_, providers=self.providers_)
        self.analysis.prepare(ctx_id=self.ctx_id, det_size=self.det_size_, det_thresh=self.det_thresh_)
        return nndeploy.base.Status.ok()
        
    def run(self):
        input_numpy = self.get_input(0).get(self)
        faces = self.analysis.get(input_numpy)
        # faces按照从左到右的顺序排列，基于bbox的x坐标进行排序
        faces = sorted(faces, key=lambda x: x.bbox[0])
        if len(faces) == 0:
            print("No face detected")
            face = faces  # 返回空列表，保持与faces一致的类型
        else:
            if self.is_one_face_:
                selected_face = min(faces, key=lambda x: x.bbox[0])
                face = [selected_face]  # 返回包含单个face的列表，保持与faces一致的类型
                # print(type(face))
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
        # json_obj["det_thresh_"] = self.det_thresh_
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
        super().set_key("nndeploy.face.InsightImageFaceId")
        super().set_desc("InsightFace Id: get face id from image")
        self.set_input_type(np.ndarray)
        self.set_input_type(list[insightface.app.common.Face])
        self.set_output_type(list[Any])
        
    def run(self):
        input_numpy = self.get_input(0).get(self)
        faces = self.get_input(1).get(self)
        face_id = []
        i = 0
        for face in faces:
            x_min, y_min, x_max, y_max = face['bbox']
            face_id.append({
                'id' : i, 
                'target' : {
                            'cv2' : input_numpy[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
        self.get_output(0).set(face_id)
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
        super().set_key("nndeploy.face.InsightVideoFaceId")
        super().set_desc("InsightFace Id: get face id from image")
        super().set_node_type(nndeploy.dag.NodeType.Input)
        
        self.video_path_ = "video.mp4"
        
        self.set_output_type(list[Any])
        
      
    def init(self):
        import nndeploy.codec as codec
        self.codec_output = nndeploy.dag.Edge("codec_output")
        self.analysis_output = nndeploy.dag.Edge("analysis_output")
        self.video_codec = codec.OpenCvVideoDecode("video_codec", [], [self.codec_output])
        self.face_analysis = InsightFaceAnalysis("face_analysis", [self.codec_output], [self.analysis_output])
        self.face_analysis.is_one_face_ = False
        self.graph = nndeploy.dag.Graph("graph", [], [self.video_codec, self.face_analysis])
        self.graph.add_node(self.video_codec)
        self.graph.add_node(self.face_analysis)
        status = self.graph.init()
        if status != nndeploy.base.Status.ok():
            print("Failed to init graph")
            return status
        return nndeploy.base.Status.ok()
      
    def run(self):
        size = self.video_codec.get_size()
        face_id = []
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
         
        
        # 使用K-means聚类算法对视频中提取的所有人脸嵌入向量进行聚类分析
        # 目的是将相似的人脸特征归为一组，从而识别视频中的不同人物
        # 返回的centroids是每个聚类的中心点，代表了视频中各个不同人物的典型特征
        centroids = find_cluster_centroids(face_embeddings)
        
        # 为每个检测到的人脸分配最匹配的聚类中心点ID
        # 遍历视频中每一帧的人脸检测结果
        for frame in frame_face_embeddings:
            # 遍历当前帧中检测到的所有人脸
            for face in frame['faces']:
                # 通过计算当前人脸嵌入向量与各个聚类中心点的相似度，
                # 找到最相似的聚类中心点，返回其索引和中心点向量
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                # 将最匹配的聚类中心点索引作为该人脸的目标聚类ID保存
                # 这样可以将同一个人在不同帧中的人脸归为同一类别
                face['target_centroid'] = closest_centroid_index  
                     
        # 为每个聚类中心点（即识别出的不同人物）创建对应的人脸ID数据结构
        for i in range(len(centroids)):
            # 为第i个聚类中心点创建一个人脸ID字典，包含该人物的唯一标识符
            face_id.append({
                'id' : i  # 人物的唯一ID，对应聚类中心点的索引
            })

            # 创建临时列表，用于存储属于当前人物（聚类中心点i）的所有帧数据
            temp = []
            # 遍历所有帧的人脸嵌入数据，筛选出属于当前聚类中心点i的人脸
            # tqdm用于显示处理进度条，方便监控大量帧数据的处理状态
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                # 为当前帧创建数据结构，只包含目标聚类为i的人脸
                # 通过列表推导式筛选出target_centroid等于i的人脸数据
                temp.append({
                    'frame': frame['frame'],  # 帧编号
                    'faces': [face for face in frame['faces'] if face['target_centroid'] == i],  # 属于聚类i的人脸列表
                    'location': frame['location']  # 帧在视频中的位置
                })

            # 将筛选后的帧数据保存到对应人物ID的数据结构中
            # 这样每个人物ID就包含了该人物在所有帧中出现的人脸数据
            face_id[i]['target_faces_in_frame'] = temp

        for map in face_id:
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
            
        self.get_output(0).set(face_id)
        
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
        self.node = InsightVideoFaceId(name, inputs, outputs)
        return self.node
      
insightvideo_face_id_node_creator = InsightVideoFaceIdCreator()
nndeploy.dag.register_node("nndeploy.face.InsightVideoFaceId", insightvideo_face_id_node_creator) 


class FaceIdMap(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.FaceIdMap")
        super().set_desc("FaceIdMap: map face id from image")
        self.set_dynamic_input(True)
        self.set_input_type(list[Any])
        self.set_input_type(list[Any])
        self.set_output_type(dict[str, list[Any]])
        
    def run(self):
        input_size = len(self.get_all_input())
        for i in range(input_size//2):
            source_face_id = self.get_input(i).get(self)
            target_face_id = self.get_input(i+1).get(self)
            face_id = []
            for source in source_face_id:
                for target in target_face_id:
                    if source['id'] == target['id']:
                        face_id.append({'source': source, 'target': target})
                        break
        
        centroids = []
        faces = []
        for map in face_id:
            if "source" in map and "target" in map:
                centroids.append(map['target']["target"]['face'].normed_embedding)
                faces.append(map['source']["target"]['face'])
        
        sim_face_id = {'source_faces': faces, 'target_embeddings': centroids}
        self.get_output(0).set(sim_face_id)
        return nndeploy.base.Status.ok()
    
class FaceIdMapCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = FaceIdMap(name, inputs, outputs)
        return self.node

face_id_map_node_creator = FaceIdMapCreator()
nndeploy.dag.register_node("nndeploy.face.FaceIdMap", face_id_map_node_creator)


class InsightFaceSwapper(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceSwapper")
        super().set_desc("InsightFace Swapper: swap face from image")
        self.set_input_type(list[insightface.app.common.Face], "source_face")
        self.set_input_type(list[insightface.app.common.Face], "target_face")
        self.set_input_type(np.ndarray, "target_frame")
        self.set_output_type(np.ndarray, "swapped_frame")
        
        self.mouth_mask_ = False
        self.show_mouth_mask_box_ = False
        self.mask_down_size_ = 0.5
        self.mask_feather_ratio_ = 8
        self.mask_size_ = 1
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
        # print(type(swapped_frame))
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        self.add_required_param("model_path_")
        self.add_required_param("providers_")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["mouth_mask_"] = self.mouth_mask_
        json_obj["show_mouth_mask_box_"] = self.show_mouth_mask_box_
        json_obj["mask_down_size_"] = self.mask_down_size_
        json_obj["mask_feather_ratio_"] = self.mask_feather_ratio_
        json_obj["mask_size_"] = self.mask_size_
        json_obj["model_path_"] = self.model_path_
        json_obj["providers_"] = self.providers_
        return json.dumps(json_obj)
      
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.mouth_mask_ = json_obj["mouth_mask_"]
        self.show_mouth_mask_box_ = json_obj["show_mouth_mask_box_"]
        self.mask_down_size_ = json_obj["mask_down_size_"]
        self.mask_feather_ratio_ = json_obj["mask_feather_ratio_"]
        self.mask_size_ = json_obj["mask_size_"]
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


class InsightFaceSwapperWithMap(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceSwapperWithMap")
        super().set_desc("InsightFace Swapper: swap face from image")
        self.set_input_type(list[insightface.app.common.Face])
        self.set_input_type(list[insightface.app.common.Face])
        self.set_input_type(np.ndarray)
        self.set_input_type(dict[str, list[Any]])
        self.set_output_type(np.ndarray)
        
        self.mouth_mask_ = False
        self.show_mouth_mask_box_ = False
        self.mask_down_size_ = 0.5
        self.mask_feather_ratio_ = 8
        self.mask_size_ = 1
        self.model_path_ = "inswapper_128_fp16.onnx"
        self.providers_ = ["CPUExecutionProvider"]
        
    def init(self):
        self.swapper = insightface.model_zoo.get_model(self.model_path_, providers=self.providers_)
        return nndeploy.base.Status.ok()
    
    def run(self):
        source_face = self.get_input(0).get(self)
        target_face = self.get_input(1).get(self)
        temp_frame = self.get_input(2).get(self)
        map = self.get_input(3).get(self)
        
        swapped_frame = temp_frame.copy()
        
        if len(target_face) == 0:
            self.get_output(0).set(swapped_frame)
            return nndeploy.base.Status.ok()
        
        detected_faces_centroids = []
        for face in target_face:
            detected_faces_centroids.append(face.normed_embedding)
        # print(map)
        for i, target_embedding in enumerate(map['target_embeddings']):
            closest_centroid_index, _ = find_closest_centroid(detected_faces_centroids, target_embedding)
            # print(closest_centroid_index)
            # print(type(target_face[closest_centroid_index]))
            # print(type(source_face[0]))
            if closest_centroid_index >= 0:
                swapped_frame = self.swapper.get(swapped_frame, target_face[closest_centroid_index], source_face[0], paste_back=True)
                    
                if self.mouth_mask_:
                    face_mask = create_face_mask(target_face[closest_centroid_index], temp_frame)

                     # Create the mouth mask
                    mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
                        create_lower_mouth_mask(target_face[closest_centroid_index], temp_frame, self.mask_down_size_, self.mask_size_)
                    )

                    # Apply the mouth area
                    swapped_frame = apply_mouth_area(
                        swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon, self.mask_feather_ratio_
                    )

                    if self.show_mouth_mask_box_:
                        mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                        swapped_frame = draw_mouth_mask_visualization(
                            swapped_frame, target_face[closest_centroid_index], mouth_mask_data, self.mask_feather_ratio_
                        )
        self.get_output(0).set(swapped_frame)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["mouth_mask_"] = self.mouth_mask_
        json_obj["show_mouth_mask_box_"] = self.show_mouth_mask_box_
        json_obj["mask_down_size_"] = self.mask_down_size_
        json_obj["mask_feather_ratio_"] = self.mask_feather_ratio_
        json_obj["mask_size_"] = self.mask_size_
        json_obj["model_path_"] = self.model_path_
        json_obj["providers_"] = self.providers_
        return json.dumps(json_obj)
      
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.mouth_mask_ = json_obj["mouth_mask_"]
        self.show_mouth_mask_box_ = json_obj["show_mouth_mask_box_"]
        self.mask_down_size_ = json_obj["mask_down_size_"]
        self.mask_feather_ratio_ = json_obj["mask_feather_ratio_"]
        self.mask_size_ = json_obj["mask_size_"]
        self.model_path_ = json_obj["model_path_"]
        self.providers_ = json_obj["providers_"]
        return super().deserialize(target)
      
class InsightFaceSwapperWithMapCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightFaceSwapperWithMap(name, inputs, outputs)
        return self.node
      
insightface_swapper_with_map_node_creator = InsightFaceSwapperWithMapCreator()
nndeploy.dag.register_node("nndeploy.face.InsightFaceSwapperWithMap", insightface_swapper_with_map_node_creator)  

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
      
      
      
        
        