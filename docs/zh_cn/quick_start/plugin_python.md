
# Python插件开发手册

## 插件开发简介

在nndeploy框架中，插件（plugin）是一种可以组合调用的功能模块，通过Python实现的插件继承自`nndeploy.dag.Node`基类。插件通过DAG（有向无环图）进行组织和调用，用于执行开发者自定义的前/后处理逻辑、推理等等。

插件设计的目的是：

- **模块化**：将特定的逻辑封装为节点，易于组合和替换
- **解耦性**：将插件与调度模块进行解耦
- **可扩展性**：开发者可以轻松接入新的算法和数据处理流程
- **跨语言互操作**：Python插件可与C++插件无缝集成

### 什么是DAG

nndeploy的执行核心是有向无环图（DAG），图由以下两个基本组件组成：

- **节点（Node）**：代表基本的计算或功能单元，可以是推理、调度、前后处理等等
- **边（Edge）**：节点之间的连接通道，用于传递张量、图像、文本等数据对象

图在运行时根据节点输入输出边判断节点的执行顺序，自动调度各个节点执行。

### 插件在流水线中的位置

以GFPGAN人脸修复为例，推理流程大致如下：

```
输入图像 → GFPGAN节点 → 修复后图像
```

以目标检测为例，推理流程包含：

```
输入图像 → 预处理节点 → 推理节点 → 后处理节点 → 检测结果
```

通过这些插件的组合，我们可以构建完整的AI算法部署流程。

## Python插件编写基础

nndeploy中的Python插件本质上是自定义的DAG节点，继承自`nndeploy.dag.Node`。要实现一个Python插件，一般需要完成以下步骤：

### 1. 定义节点类

```python
import numpy as np
import json
import nndeploy.dag
import nndeploy.base

class MyCustomNode(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        # 设置节点key，与注册的节点类型名一致，必须要设置
        super().set_key("nndeploy.example.MyCustomNode")
        # 设置节点描述
        super().set_desc("自定义节点示例")
        # 设置输入类型，如果该节点有输入，则必须要设置
        self.set_input_type(np.ndarray)
        # 设置输出类型，如果该节点有输出，则必须要设置
        self.set_output_type(np.ndarray)
        
        # 节点参数定义
        self.my_param = 1.0
        
    def init(self):
        """节点初始化方法，在图初始化时调用"""
        # 在这里进行模型加载、资源初始化等操作
        return nndeploy.base.Status.ok()
        
    def run(self):
        """节点执行方法，包含主要的计算逻辑"""
        # 获取输入数据
        input_edge = self.get_input(0)
        input_data = input_edge.get(self)
        
        # 执行计算逻辑
        output_data = self.process_data(input_data)
        
        # 设置输出数据
        output_edge = self.get_output(0)
        output_edge.set(output_data)
        
        return nndeploy.base.Status.ok()
        
    def process_data(self, input_data):
        """自定义数据处理逻辑"""
        # 在这里实现具体的算法逻辑
        return input_data * self.my_param
        
    def serialize(self):
        """序列化节点参数，用于保存和加载配置"""
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["my_param"] = self.my_param
        return json.dumps(json_obj)
        
    def deserialize(self, target: str):
        """反序列化节点参数"""
        json_obj = json.loads(target)
        self.my_param = json_obj["my_param"]
        return super().deserialize(target)
```

### 2. 创建节点创建器

```python
class MyCustomNodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """创建节点实例"""
        self.node = MyCustomNode(name, inputs, outputs)
        return self.node
```

### 3. 注册节点类型

```python
# 创建节点创建器实例
my_custom_node_creator = MyCustomNodeCreator()

# 注册节点，第一个参数是节点的唯一标识符，需要与节点类中set_key()设置的标识符保持一致
nndeploy.dag.register_node("nndeploy.example.MyCustomNode", my_custom_node_creator)
```

### 4. 节点中的输入输出访问

#### 获取输入数据

```python
def run(self):
    # 获取第一个输入边
    input_edge = self.get_input(0)
    # 从输入边获取数据
    input_data = input_edge.get(self)
    
    # 如果有多个输入
    input_edge_1 = self.get_input(1)
    input_data_1 = input_edge_1.get(self)
```

#### 设置输出数据

```python
def run(self):
    # 处理后的输出数据
    output_data = self.process_data(input_data)
    
    # 获取输出边并设置数据
    output_edge = self.get_output(0)
    output_edge.set(output_data)
    
    # 如果有多个输出
    output_edge_1 = self.get_output(1)
    output_edge_1.set(output_data_1)
```

### 5. 参数管理

#### 前端展示参数

```python
class MyCustomNode(nndeploy.dag.Node):
    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        # ... 其他初始化代码 ...
        
        # 前端需要展示的参数
        self.threshold = 0.5  # float类型参数
        self.enable_flag = True  # bool类型参数
        self.model_path = "model.onnx"  # 字符串类型参数
        self.class_names = ["person", "car", "bike"]  # list类型参数
        self.config = {"width": 640, "height": 640}  # dict类型参数
        
    def serialize(self):
        """序列化前端参数"""
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["threshold"] = self.threshold
        json_obj["enable_flag"] = self.enable_flag
        json_obj["model_path"] = self.model_path
        json_obj["class_names"] = self.class_names
        json_obj["config"] = self.config
        return json.dumps(json_obj)
        
    def deserialize(self, target: str):
        """反序列化前端参数"""
        json_obj = json.loads(target)
        self.threshold = json_obj.get("threshold", 0.5)
        self.enable_flag = json_obj.get("enable_flag", True)
        self.model_path = json_obj.get("model_path", "model.onnx")
        self.class_names = json_obj.get("class_names", [])
        self.config = json_obj.get("config", {})
        return super().deserialize(target)
```

#### 必需参数管理

```python
def serialize(self):
    # 添加必需参数，前端会进行验证
    self.add_required_param("model_path")
    json_str = super().serialize()
    json_obj = json.loads(json_str)
    json_obj["model_path"] = self.model_path
    return json.dumps(json_obj)
```

## 实际开发案例

### 案例一：图像灰度化节点

```python
import numpy as np
import json
import nndeploy.dag
import nndeploy.base

class GrayScaleNode(nndeploy.dag.Node):
    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.image.GrayScale")
        super().set_desc("将彩色图像转换为灰度图像")
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
        # 灰度化权重参数
        self.weights = [0.114, 0.587, 0.299]  # BGR权重
        
    def run(self):
        input_edge = self.get_input(0)
        input_image = input_edge.get(self)
        
        # BGR转灰度
        gray = np.dot(input_image[...,:3], self.weights)
        gray = gray.astype(np.uint8)
        
        output_edge = self.get_output(0)
        output_edge.set(gray)
        return nndeploy.base.Status.ok()
        
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["weights"] = self.weights
        return json.dumps(json_obj)
        
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.weights = json_obj.get("weights", [0.114, 0.587, 0.299])
        return super().deserialize(target)

# 创建器和注册
class GrayScaleNodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = GrayScaleNode(name, inputs, outputs) # 必须这样写
        return self.node

grayscale_node_creator = GrayScaleNodeCreator()
nndeploy.dag.register_node("nndeploy.image.GrayScale", grayscale_node_creator)
```

### 案例二：GFPGAN人脸修复节点

基于实际的GFPGAN实现：

```python
import gfpgan
import numpy as np
import json
import nndeploy.base
import nndeploy.device
import nndeploy.dag

class GFPGAN(nndeploy.dag.Node):
    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.gan.GFPGAN")
        super().set_desc("GFPGAN: Make faces clearer")
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
        # 模型参数
        self.model_path_ = "GFPGANv1.4.pth"
        self.upscale_ = 1
        self.device_, _ = nndeploy.device.get_available_device()
        
    def init(self):
        """初始化GFPGAN模型"""
        self.gfpgan = gfpgan.GFPGANer(
            self.model_path_, 
            upscale=self.upscale_, 
            device=self.device_
        )
        return nndeploy.base.Status.ok()
        
    def run(self):
        """执行人脸修复"""
        input_edge = self.get_input(0)
        input_image = input_edge.get(self)
        
        # 执行人脸修复
        _, _, enhanced_image = self.gfpgan.enhance(input_image, paste_back=True)
        
        # 输出结果
        self.get_output(0).set(enhanced_image)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        """序列化参数"""
        self.add_required_param("model_path_")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["model_path_"] = self.model_path_
        json_obj["upscale_"] = self.upscale_
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        """反序列化参数"""
        json_obj = json.loads(target)
        self.model_path_ = json_obj["model_path_"]
        self.upscale_ = json_obj["upscale_"]
        return super().deserialize(target)

# 创建器和注册
class GFPGANCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = GFPGAN(name, inputs, outputs)
        return self.node
    
gfpgan_node_creator = GFPGANCreator()
nndeploy.dag.register_node("nndeploy.gan.GFPGAN", gfpgan_node_creator)
```

### 案例三：复杂的人脸分析节点

基于InsightFace的人脸分析实现：

```python
import numpy as np
import json
import insightface
import nndeploy.base
import nndeploy.dag

class InsightFaceAnalysis(nndeploy.dag.Node):
    def __init__(self, name, inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.face.InsightFaceAnalysis")
        super().set_desc("InsightFace Analysis: get face analysis from image")
        self.set_input_type(np.ndarray)
        self.set_output_type(list[insightface.app.common.Face])
        
        # 分析参数
        self.insightface_name_ = "buffalo_l"
        self.providers_ = ["CPUExecutionProvider"]
        self.is_one_face_ = True
        self.ctx_id = 0
        self.det_size_ = (640, 640)
        self.det_thresh_ = 0.5
        
    def init(self):
        """初始化人脸分析模型"""
        self.analysis = insightface.app.FaceAnalysis(
            name=self.insightface_name_, 
            providers=self.providers_
        )
        self.analysis.prepare(
            ctx_id=self.ctx_id, 
            det_size=self.det_size_, 
            det_thresh=self.det_thresh_
        )
        return nndeploy.base.Status.ok()
        
    def run(self):
        """执行人脸分析"""
        input_image = self.get_input(0).get(self)
        faces = self.analysis.get(input_image)
        
        # 按照从左到右的顺序排列，基于bbox的x坐标进行排序
        faces = sorted(faces, key=lambda x: x.bbox[0])
        
        if len(faces) == 0:
            print("No face detected")
            result_faces = faces  # 返回空列表
        else:
            if self.is_one_face_:
                # 选择最左边的人脸
                selected_face = min(faces, key=lambda x: x.bbox[0])
                result_faces = [selected_face]
            else:
                result_faces = faces  # 返回所有人脸
                
        self.get_output(0).set(result_faces)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        """序列化参数"""
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["insightface_name_"] = self.insightface_name_
        json_obj["providers_"] = self.providers_
        json_obj["is_one_face_"] = self.is_one_face_
        json_obj["ctx_id"] = self.ctx_id
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        """反序列化参数"""
        json_obj = json.loads(target)
        self.insightface_name_ = json_obj["insightface_name_"]
        self.providers_ = json_obj["providers_"]
        self.is_one_face_ = json_obj["is_one_face_"]
        self.ctx_id = json_obj["ctx_id"]
        return super().deserialize(target)

# 创建器和注册
class InsightFaceAnalysisCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InsightFaceAnalysis(name, inputs, outputs)
        return self.node

insightface_analysis_creator = InsightFaceAnalysisCreator()
nndeploy.dag.register_node("nndeploy.face.InsightFaceAnalysis", insightface_analysis_creator)
```

## 高级用法

### 创建子图

当插件涉及更复杂的功能逻辑时，可以通过创建子图（继承`nndeploy.dag.Graph`）来组织多个节点：

```python
import nndeploy.dag
import nndeploy.base

class YoloPyGraph(nndeploy.dag.Graph):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key(type(self).__name__)
        self.set_input_type(np.ndarray)
        self.set_output_type(nndeploy.detect.DetectResult)
        self.pre = self.create_node("nndeploy::preprocess::CvtResizeNormTrans", "pre")
        self.infer = self.create_node("nndeploy::infer::Infer", "infer")
        self.post = self.create_node("nndeploy::detect::YoloPostProcess", "post")

    # 动态图方式    
    def forward(self, inputs: [nndeploy.dag.Edge]):
        pre_outputs = self.pre(inputs)
        infer_outputs = self.infer(pre_outputs)
        post_outputs = self.post(infer_outputs)
        return post_outputs
    
    # 静态图方式
    def make(self, pre_desc, infer_desc, post_desc):
        self.set_node_desc(self.pre, pre_desc)
        self.set_node_desc(self.infer, infer_desc)
        self.set_node_desc(self.post, post_desc)
        return nndeploy.base.StatusCode.Ok
        
    def default_param(self):
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

        return nndeploy.base.StatusCode.Ok
    
    def set_inference_type(self, inference_type):
        self.infer.set_inference_type(inference_type)
        
    def set_infer_param(self, device_type, model_type, is_path, model_value):
        param = self.infer.get_param()
        param.device_type_ = device_type
        param.model_type_ = model_type 
        param.is_path_ = is_path
        param.model_value_ = model_value
        return nndeploy.base.StatusCode.Ok

    def set_src_pixel_type(self, pixel_type):
        param = self.pre.get_param()
        param.src_pixel_type_ = pixel_type
        return nndeploy.base.StatusCode.Ok

    def set_score_threshold(self, score_threshold):
        param = self.post.get_param()
        param.score_threshold_ = score_threshold
        return nndeploy.base.StatusCode.Ok

    def set_nms_threshold(self, nms_threshold):
        param = self.post.get_param()
        param.nms_threshold_ = nms_threshold
        return nndeploy.base.StatusCode.Ok

    def set_num_classes(self, num_classes):
        param = self.post.get_param()
        param.num_classes_ = num_classes
        return nndeploy.base.StatusCode.Ok

    def set_model_hw(self, model_h, model_w):
        param = self.post.get_param()
        param.model_h_ = model_h
        param.model_w_ = model_w
        return nndeploy.base.StatusCode.Ok

    def set_version(self, version):
        param = self.post.get_param()
        param.version_ = version
        return nndeploy.base.StatusCode.Ok
        

class YoloPyGraphCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = YoloPyGraph(name, inputs, outputs)
        return self.node
    

yolo_py_graph_creator = YoloPyGraphCreator()
nndeploy.dag.register_node("nndeploy.detect.YoloPyGraph", yolo_py_graph_creator)
```

## 最佳实践

### 1. 命名规范

- 节点key使用层次化命名：`nndeploy.模块.功能`
- 类名使用驼峰命名法：`MyCustomNode`
- 参数使用下划线命名：`model_path_`

### 2. 类型安全

```python
# 明确指定输入输出类型
self.set_input_type(np.ndarray)
self.set_output_type(dict)

# 在运行时进行类型检查
def run(self):
    input_data = self.get_input(0).get(self)
    if not isinstance(input_data, np.ndarray):
        print("Input type error: expected np.ndarray")
        return nndeploy.base.Status.error()
```

### 3. 资源管理

```python
def init(self):
    """在init中初始化资源"""
    self.model = load_model(self.model_path_)
    return nndeploy.base.Status.ok()
    
def deinit(self):
    """在deinit中清理资源"""
    if hasattr(self, 'model'):
        del self.model
    return super().deinit()
```

### 4. 调试支持

```python
def run(self):
    if self.get_debug_flag():
        print(f"Node {self.get_name()} is running")
        print(f"Input shape: {input_data.shape}")
        
    # 执行逻辑
    output_data = self.process_data(input_data)
    
    if self.get_debug_flag():
        print(f"Output shape: {output_data.shape}")
        
    self.get_output(0).set(output_data)
    return nndeploy.base.Status.ok()
```

## 插件部署和使用

### 1. 文件组织

```
my_plugin/
├── __init__.py
├── my_node.py
└── README.md
```

### 2. 模块导入

```python
# __init__.py
from .yolo import YoloPyGraph
```

### 3. 在应用中使用

```python

import nndeploy.dag
import nndeploy.detect
class YoloDemo(nndeploy.dag.Graph):
    def __init__(self, name = "", inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("YoloDemo")
        self.set_output_type(nndeploy.detect.DetectResult)
        self.decodec = nndeploy.codec.OpenCvImageDecode("decodec")
        self.yolo = nndeploy.detect.YoloPyGraph("yolo")
        self.drawbox = nndeploy.detect.DrawBox("drawbox")
        self.encodec = nndeploy.codec.OpenCvImageEncode("encodec")
        
    def forward(self, inputs: [nndeploy.dag.Edge] = []):
        decodec_outputs = self.decodec(inputs)
        yolo_outputs = self.yolo(decodec_outputs)
        drawbox_outputs = self.drawbox([decodec_outputs[0], yolo_outputs[0]])
        self.encodec(drawbox_outputs)
        return yolo_outputs
       
    def get_yolo(self):
        return self.yolo
    
    def set_size(self, size):
        self.decodec.set_size(size)
    
    def set_input_path(self, path):
        self.decodec.set_path(path)
        
    def set_output_path(self, path):
        self.encodec.set_path(path)

```