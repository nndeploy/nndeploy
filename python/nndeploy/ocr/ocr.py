import nndeploy._nndeploy_internal as _C
OCRResult = _C.ocr.OCRResult
DrawDetectorBox = _C.ocr.DrawDetectorBox
RotateCropImage = _C.ocr.RotateCropImage
RotateImage180 = _C.ocr.RotateImage180
try:
    DetectorPreProcessParam = _C.ocr.DetectorPreProcessParam
    DetectorPreProcess = _C.ocr.DetectorPreProcess
    DetectorPostParam = _C.ocr.DetectorPostParam
    DetectorPostProcess = _C.ocr.DetectorPostProcess
    DetectorGraph = _C.ocr.DetectorGraph
    
    ClassifierPreProcessParam = _C.ocr.ClassifierPreProcessParam
    ClassifierPreProcess = _C.ocr.ClassifierPreProcess
    ClassifierPostParam = _C.ocr.ClassifierPostParam
    ClassifierPostProcess = _C.ocr.ClassifierPostProcess
    ClassifierGraph = _C.ocr.ClassifierGraph

    RecognizerPreProcessParam = _C.ocr.RecognizerPreProcessParam
    RecognizerPreProcess = _C.ocr.RecognizerPreProcess
    RecognizerPostParam = _C.ocr.RecognizerPostParam
    RecognizerPostProcess = _C.ocr.RecognizerPostProcess
    RecognizerGraph = _C.ocr.RecognizerGraph

    
    
    OcrText = _C.ocr.OcrText
    PrintOcrNode = _C.ocr.PrintOcrNode
    PrintOcrNodeParam = _C.ocr.PrintOcrNodeParam

    
except:
    pass


# class DetectorPyGraph(nndeploy.dag.Graph):
#     def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
#         super().__init__(name, inputs, outputs)
#         self.set_key(type(self).__name__)
#         self.set_input_type(np.ndarray)
#         self.set_output_type(nndeploy.detect.DetectResult)
#         self.pre = self.create_node("nndeploy::preprocess::CvtResizeNormTrans", "pre")
#         self.infer = self.create_node("nndeploy::infer::Infer", "infer")
#         self.post = self.create_node("nndeploy::detect::YoloPostProcess", "post")
        
#     def forward(self, inputs: [nndeploy.dag.Edge]):
#         pre_outputs = self.pre(inputs)
#         infer_outputs = self.infer(pre_outputs)
#         post_outputs = self.post(infer_outputs)
#         return post_outputs
    
#     def make(self, pre_desc, infer_desc, post_desc):
#         self.set_node_desc(self.pre, pre_desc)
#         self.set_node_desc(self.infer, infer_desc)
#         self.set_node_desc(self.post, post_desc)
#         return nndeploy.base.StatusCode.Ok
        
#     def default_param(self):
#         pre_param = self.pre.get_param()
#         pre_param.src_pixel_type_ = nndeploy.base.PixelType.BGR
#         pre_param.dst_pixel_type_ = nndeploy.base.PixelType.RGB
#         pre_param.interp_type_ = nndeploy.base.InterpType.Linear
#         pre_param.h_ = 640
#         pre_param.w_ = 640

#         post_param = self.post.get_param()
#         post_param.score_threshold_ = 0.5
#         post_param.nms_threshold_ = 0.45
#         post_param.num_classes_ = 80
#         post_param.model_h_ = 640
#         post_param.model_w_ = 640
#         post_param.version_ = 11

#         return nndeploy.base.StatusCode.Ok
    
#     def set_inference_type(self, inference_type):
#         self.infer.set_inference_type(inference_type)
        
#     def set_infer_param(self, device_type, model_type, is_path, model_value):
#         param = self.infer.get_param()
#         param.device_type_ = device_type
#         param.model_type_ = model_type 
#         param.is_path_ = is_path
#         param.model_value_ = model_value
#         return nndeploy.base.StatusCode.Ok

#     def set_src_pixel_type(self, pixel_type):
#         param = self.pre.get_param()
#         param.src_pixel_type_ = pixel_type
#         return nndeploy.base.StatusCode.Ok

#     def set_score_threshold(self, score_threshold):
#         param = self.post.get_param()
#         param.score_threshold_ = score_threshold
#         return nndeploy.base.StatusCode.Ok

#     def set_nms_threshold(self, nms_threshold):
#         param = self.post.get_param()
#         param.nms_threshold_ = nms_threshold
#         return nndeploy.base.StatusCode.Ok

#     def set_num_classes(self, num_classes):
#         param = self.post.get_param()
#         param.num_classes_ = num_classes
#         return nndeploy.base.StatusCode.Ok

#     def set_model_hw(self, model_h, model_w):
#         param = self.post.get_param()
#         param.model_h_ = model_h
#         param.model_w_ = model_w
#         return nndeploy.base.StatusCode.Ok

#     def set_version(self, version):
#         param = self.post.get_param()
#         param.version_ = version
#         return nndeploy.base.StatusCode.Ok
        

# class YoloPyGraphCreator(nndeploy.dag.NodeCreator):
#     def __init__(self):
#         super().__init__()
        
#     def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
#         self.node = YoloPyGraph(name, inputs, outputs)
#         return self.node
    

# yolo_py_graph_creator = YoloPyGraphCreator()
# nndeploy.dag.register_node("nndeploy.detect.YoloPyGraph", yolo_py_graph_creator)