
# 
# The following commands are for installing necessary Python libraries to handle and optimize ONNX cpu_models
# pip3 install ultralytics
# pip3 install nndeploy

from ultralytics import YOLO  # Import YOLO cpu_model class from ultralytics library
import time

import nndeploy.base
import nndeploy.device
import nndeploy.dag
import nndeploy.detect
import nndeploy.codec

import sys

# benchmark count
count = 100

# yolo 官方 cuda fp32 耗时(RTX 3060)
# warmup
# cuda_model = YOLO("yolo11s.pt")
# yolo_cuda_sum_time = 0
# for i in range(count):
#     start_time = time.time()
#     results = cuda_model("/home/always/github/public/nndeploy/docs/image/demo/detect/sample.jpg", save=True, project="build", name="exp1", device="cuda", imgsz=640, conf=0.5, iou=0.45, half=False)
#     end_time = time.time()
#     elapsed_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
#     yolo_cuda_sum_time += elapsed_ms
# print(f"[YOLOv11s CUDA FP32 Benchmark] Average inference time: {yolo_cuda_sum_time / count:.2f} ms")

# # yolo 官方 cpu fp32 耗时
# # warmup
# cpu_model = YOLO("yolo11s.pt")
# yolo_cpu_sum_time = 0
# for i in range(count):
#     start_time = time.time()
#     results = cpu_model("/home/always/github/public/nndeploy/docs/image/demo/detect/sample.jpg", save=True, project="build", name="exp1", device="cpu", imgsz=640, conf=0.5, iou=0.45, half=False)
#     end_time = time.time()
#     elapsed_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
#     yolo_cpu_sum_time += elapsed_ms
# print(f"[YOLOv11s CPU FP32 Benchmark] Average inference time: {yolo_cpu_sum_time / count:.2f} ms")

# nndeploy 
class YoloPyGraph(nndeploy.dag.Graph):
    """YOLO graph implementation for deploy pipeline, using PyTorch-like style"""
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key(type(self).__name__)
        self.set_desc("YoloPyGraph")
        self.set_input_type(np.ndarray)
        self.set_output_type(nndeploy.detect.DetectResult)
        # Define modules in a PyTorch-like style: preprocess, inference, postprocess
        self.pre = self.create_node("nndeploy::preprocess::CvtResizeNormTrans", "pre")  # Preprocessing module
        self.infer = self.create_node("nndeploy::infer::Infer", "infer")  # Inference module
        self.post = self.create_node("nndeploy::detect::YoloPostProcess", "post")  # Postprocessing module
        
    def forward(self, inputs: [nndeploy.dag.Edge]):
        """Forward pass, similar to PyTorch's forward method"""
        # Forward pipeline: preprocess -> infer -> postprocess
        pre_outputs = self.pre(inputs)  # Preprocessing
        infer_outputs = self.infer(pre_outputs)  # Inference
        post_outputs = self.post(infer_outputs)  # Postprocessing
        return post_outputs
            
    def default_param(self):
        """Set default parameters for preprocessing"""
        pre_param = self.pre.get_param()
        pre_param.src_pixel_type_ = nndeploy.base.PixelType.BGR
        pre_param.dst_pixel_type_ = nndeploy.base.PixelType.RGB
        pre_param.interp_type_ = nndeploy.base.InterpType.Linear
        pre_param.h_ = 640
        pre_param.w_ = 640

        post_param = self.post.get_param()
        post_param.score_threshold_ = 0.5
        post_param.nms_threshold_ = 0.45
        post_param.num_classes_ = 80
        post_param.model_h_ = 640
        post_param.model_w_ = 640
        post_param.version_ = 11
    
    def set_inference_type(self, inference_type):
        """Set inference backend type"""
        self.infer.set_inference_type(inference_type)
        
    def set_infer_param(self, device_type, model_type, is_path, model_value):
        """Configure inference parameters"""
        param = self.infer.get_param()
        param.device_type_ = device_type
        param.model_type_ = model_type 
        param.is_path_ = is_path
        param.model_value_ = model_value
        return nndeploy.base.StatusCode.Ok

    def set_version(self, version):
        """Set YOLO version for postprocessing"""
        param = self.post.get_param()
        param.version_ = version
        return nndeploy.base.StatusCode.Ok

class YoloPyGraphCreator(nndeploy.dag.NodeCreator):
    """Factory class for creating YoloPyGraph instances"""
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = YoloPyGraph(name, inputs, outputs)
        return self.node
    
# Register the graph creator
yolo_py_graph_creator = YoloPyGraphCreator()
nndeploy.dag.register_node("nndeploy.detect.YoloPyGraph", yolo_py_graph_creator)

class YoloPyDemo(nndeploy.dag.Graph):
    """End-to-end YOLO demo implementation, using PyTorch-like style"""
    def __init__(self, name = "", inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key(type(self).__name__)
        self.set_desc("YoloPyDemo")
        self.set_output_type(nndeploy.detect.DetectResult)
        # Create pipeline components
        self.decodec = nndeploy.codec.OpenCvImageDecode("decodec")  # Image decoder
        self.yolo = nndeploy.detect.YoloPyGraph("yolo")  # YOLO graph
        self.drawbox = nndeploy.detect.DrawBox("drawbox")  # Bounding box drawer
        self.encodec = nndeploy.codec.OpenCvImageEncode("encodec")  # Image encoder
        
    def forward(self):
        """Forward pass, similar to PyTorch's forward method"""
        decodec_outputs = self.decodec()
        yolo_outputs = self.yolo(decodec_outputs)
        drawbox_outputs = self.drawbox([decodec_outputs[0], yolo_outputs[0]])
        self.encodec(drawbox_outputs)
        return yolo_outputs
       
    def get_yolo(self):
        """Get YOLO graph instance"""
        return self.yolo
    
    def set_size(self, size):
        self.decodec.set_size(size)
        
    def set_input_path(self, path):
        """Set input image path"""
        self.decodec.set_path(path)
        
    def set_output_path(self, path):
        """Set output image path"""
        self.encodec.set_path(path)
        

if __name__ == "__main__":
    # Main execution for YOLO demo
    yolo_py_demo = YoloPyDemo("yolo_py_demo")

    # Configure YOLO parameters
    yolo = yolo_py_demo.get_yolo()
    yolo.default_param()
    yolo.set_inference_type(nndeploy.base.InferenceType.OnnxRuntime)
    yolo.set_infer_param(nndeploy.base.DeviceType("x86"), nndeploy.base.ModelType.Onnx, True, ["/home/lds/modelscope/nndeploy/detect/yolo11s.sim.onnx"])

    # Set input/output paths
    yolo_py_demo.set_input_path("/home/always/github/public/nndeploy/docs/image/demo/detect/sample.jpg")
    yolo_py_demo.set_output_path("/home/always/github/public/nndeploy/build/yolo_python_demo.jpg")

    # Run end-to-end demo
    output = yolo_py_demo()
    # Save results to file
    yolo_py_demo.save_file("/home/always/github/public/nndeploy/build/yolo_python_demo.json")
    
    # onnxruntime YOLO model through json file
    # json_model = YoloPyDemo("json_model")
    json_model = nndeploy.dag.Graph("json_model")
    json_model.load_file("/home/always/github/public/nndeploy/resources/workflow/yolo.json")
    json_model.set_time_profile_flag(True)
    status = json_model.init()
    if status != nndeploy.base.StatusCode.Ok:
        raise RuntimeError(f"init failed: {status}")
    json_model_sum_time = 0
    # warmup
    # output = json_model()
    for i in range(count):
        start_time = time.time()
        status = json_model.run()
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
        json_model_sum_time += elapsed_ms
    print(f"[YOLOv11s Python Benchmark OnnxRuntime] Average inference time: {json_model_sum_time / count:.2f} ms")
    nndeploy.base.time_profiler_print("json_model")
    
    json_model = YoloPyDemo("json_model")
    json_model.load_file("/home/always/github/public/nndeploy/build/yolo_python_demo.json")
    json_model.set_parallel_type(nndeploy.base.ParallelType.Pipeline)
    status = json_model.init()
    if status != nndeploy.base.StatusCode.Ok:
        raise RuntimeError(f"init failed: {status}")
    json_model_sum_time = 0
    json_model.set_size(count)
    start_time = time.time()
    for i in range(count):
        status = json_model.run()
    json_model.synchronize()
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    json_model_sum_time += elapsed_ms
    print(f"[YOLOv11s Python Benchmark OnnxRuntime Pipeline] Average inference time: {json_model_sum_time / count:.2f} ms")
    
    # yolo_py_demo = YoloPyDemo("yolo_py_demo")

    # # Configure YOLO parameters
    # yolo = yolo_py_demo.get_yolo()
    # yolo.default_param()
    # yolo.set_inference_type(nndeploy.base.InferenceType.TensorRt)
    # yolo.set_infer_param(nndeploy.base.DeviceType("cuda"), nndeploy.base.ModelType.Onnx, True, ["/home/lds/modelscope/nndeploy/detect/yolo11s.sim.onnx"])
    # print("yolo_py_demo")

    # # Set input/output paths
    # yolo_py_demo.set_input_path("/home/always/github/public/nndeploy/docs/image/demo/detect/sample.jpg")
    # yolo_py_demo.set_output_path("/home/always/github/public/nndeploy/build/yolo_python_demo.jpg")
    # print("yolo_py_demo")

    # # Run end-to-end demo
    # output = yolo_py_demo()
    # # Save results to file
    # yolo_py_demo.save_file("/home/always/github/public/nndeploy/build/yolo_python_demo_openvino.json")
    # print("yolo_py_demo")
    
    
    # # openvino YOLO model through json file
    # json_model = YoloPyDemo("json_model")
    # json_model.load_file("/home/always/github/public/nndeploy/build/yolo_python_demo_openvino.json")
    # status = json_model.init()
    # if status != nndeploy.base.StatusCode.Ok:
    #     raise RuntimeError(f"init failed: {status}")
    # json_model_sum_time = 0
    # # warmup
    # # output = json_model()
    # for i in range(count):
    #     start_time = time.time()
    #     status = json_model.run()
    #     end_time = time.time()
    #     elapsed_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    #     json_model_sum_time += elapsed_ms
    # print(f"[YOLOv11s Python Benchmark OpenVino] Average inference time: {json_model_sum_time / count:.2f} ms")
    
    # json_model = YoloPyDemo("json_model")
    # json_model.load_file("/home/always/github/public/nndeploy/build/yolo_python_demo_openvino.json")
    # json_model.set_parallel_type(nndeploy.base.ParallelType.Pipeline)
    # status = json_model.init()
    # if status != nndeploy.base.StatusCode.Ok:
    #     raise RuntimeError(f"init failed: {status}")
    # json_model_sum_time = 0
    # json_model.set_size(count)
    # start_time = time.time()
    # for i in range(count):
    #     status = json_model.run()
    # json_model.synchronize()
    # end_time = time.time()
    # elapsed_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    # json_model_sum_time += elapsed_ms
    # print(f"[YOLOv11s Python Benchmark OpenVino Pipeline] Average inference time: {json_model_sum_time / count:.2f} ms")
    
    
    
    
    
