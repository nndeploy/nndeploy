### nndeploy rknn yolov5 demo使用说明

测试环境

rknn_toolkit1 - 3399pro
rknn_toolkit2 - 3588

#### step1

修改 ``` cmake/config_linux.cmake``` 

将不需要的选项置为OFF

例如

```
set(ENABLE_NNDEPLOY_DEVICE_CUDA OFF)
set(ENABLE_NNDEPLOY_DEVICE_CUDNN OFF)
set(ENABLE_NNDEPLOY_INFERENCE_NCNN OFF)
...
```

根据使用的设备将相应的选项设为rknn库的路径，另一个设为OFF
```
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 /path/to/rknn_lib_folder)
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 OFF)
```

rknn库的文件夹结构如下

```
rknn_lib_folder
	include
		rknn_api.h
	lib
		librknn_api.so
```

然后将cmake/config_linux.cmake移动到build文件夹下并重命名为config.cmake

在build目录下编译

```
cmake ..
make -j4
```

#### step2

下载模型和yolo后处理代码，替换项目中的源文件

```
链接：https://pan.baidu.com/s/1PzD4WUL3wZCS0968pGUuFA 
提取码：l47m 
```

修改输入输出名称

3399改为

```
  dag::Edge *infer_0 = graph->createEdge("Sigmoid_Sigmoid_199/out0_0");
  dag::Edge *infer_1 = graph->createEdge("Sigmoid_Sigmoid_201/out0_1");
  dag::Edge *infer_2 = graph->createEdge("Sigmoid_Sigmoid_203/out0_2");
```

3588改为

```
  dag::Edge *infer_input = graph->createEdge("image");
  dag::Edge *infer_0 = graph->createEdge("output0");
  dag::Edge *infer_1 = graph->createEdge("output1");
  dag::Edge *infer_2 = graph->createEdge("output2");
```

### step3

可以运行啦


```
demo_nndeploy_detect
--name
NNDEPLOY_YOLOV5_MULTI_OUTPUT
--inference_type
kInferenceTypeRknn
--device_type
kDeviceTypeCodeCpu:0
--model_type
kModelTypeRknn
--is_path
--model_value
/path_to_rknn_model
--input_type
kInputTypeImage
--input_path
/path_to_input_path
--output_path
/path_to_output_path
```

