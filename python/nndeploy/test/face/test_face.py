

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.face
import nndeploy.gan
import nndeploy.codec
import torch
import numpy as np

import sys


# python3 nndeploy/test/detect/test_detect.py

# 创建的所有节点都最好都是成员变量self.xxx，不要在forward中创建节点
class FaceSwapperDemo(nndeploy.dag.Graph):
    def __init__(self, name = "", inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.test.FaceSwapperDemo")
        self.set_desc("FaceSwapperDemo: swap face from image")
        self.set_output_type(np.ndarray)
        self.decodec_source = nndeploy.codec.OpenCvImageDecode("decodec_source")
        self.decodec_target = nndeploy.codec.OpenCvImageDecode("decodec_target")
        self.face_swapper = nndeploy.face.FaceSwapper("face_swapper")
        self.insightface = nndeploy.face.InsightFaceAnalysis("insightface")
        self.gfpgan = nndeploy.gan.GFPGAN("gfpgan")
        self.encodec = nndeploy.codec.OpenCvImageEncode("encodec")
    
    def forward(self):
        decodec_source_outputs = self.decodec_source()
        decodec_target_outputs = self.decodec_target()
        face_swapper_outputs = self.face_swapper([decodec_source_outputs[0], decodec_target_outputs[0]])
        gfpgan_outputs = self.gfpgan([face_swapper_outputs[0]])
        self.encodec(gfpgan_outputs)
        return gfpgan_outputs
       
    def set_size(self, size):
        self.decodec_target.set_size(size)
    
    def set_input_path_source(self, path):
        self.decodec_source.set_path(path)
        
    def set_input_path_target(self, path):
        self.decodec_target.set_path(path)
        
    def set_output_path(self, path):
        self.encodec.set_path(path)
        
    def set_face_swapper_model_path(self, model_path):
        self.face_swapper.set_face_swapper_model_path(model_path)   
        
    def set_gfpgan_model_path(self, model_path):
        self.gfpgan.model_path_ = model_path
        
        
def test_face_swapper():    
    face_swapper_demo = FaceSwapperDemo("face_swapper_demo")
    face_swapper_demo.set_face_swapper_model_path("/home/always/github/Deep-Live-Cam/models/inswapper_128_fp16.onnx")
    face_swapper_demo.set_gfpgan_model_path("/home/always/github/Deep-Live-Cam/models/GFPGANv1.4.pth")
    face_swapper_demo.set_time_profile_flag(True)
    output = face_swapper_demo.trace()
    face_swapper_demo.dump()
    face_swapper_demo.set_input_path_source("/home/always/github/Deep-Live-Cam/Always.jpg")
    face_swapper_demo.set_input_path_target("/home/always/github/Deep-Live-Cam/Chunel.jpg")
    face_swapper_demo.set_output_path("/home/always/github/public/nndeploy/build/nndeploy_Always_Chunel.jpg")
    count = 1
    face_swapper_demo.set_size(count)
    nndeploy.base.time_point_start("face_swapper_demo_python")
    for i in range(count):
        output = face_swapper_demo()
    nndeploy.base.time_point_end("face_swapper_demo_python")  
    face_swapper_demo.save_file("/home/always/github/public/nndeploy/build/face_swapper_demo_v3.json")
    nndeploy.base.time_profiler_print("face_swapper_demo")
    
     
def test_face_swapper_from_json():  
    face_swapper_demo = FaceSwapperDemo("face_swapper_demo")
    face_swapper_demo.load_file("/home/always/github/public/nndeploy/build/face_swapper_demo_v3.json")
    face_swapper_demo.set_output_path("/home/always/github/Deep-Live-Cam/nndeploy_Always_Chunel_v3.jpg")
    face_swapper_demo.set_time_profile_flag(True)
    face_swapper_demo.init()
    face_swapper_demo.dump()
    face_swapper_demo.run()
    nndeploy.base.time_point_end("face_swapper_demo_python")
    
    
if __name__ == "__main__":
    test_face_swapper()
    # test_face_swapper_from_json()
    
        
        
        
        
