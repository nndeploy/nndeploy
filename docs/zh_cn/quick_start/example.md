# 示例工程

## 从源码编译

参考编译文档，[编译文档链接](./build.md)

## 基于DAG的模型部署演示示例（采用默认config.cmake即可编译成功）

### Windows 下运行 demo_nndeploy_dag
```shell
cd PATH/nndeploy/build/install/bin
.\demo_nndeploy_dag.exe
```

### Linux 下运行 demo_nndeploy_dag
```shell
cd PATH/nndeploy/build/install/lib
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
./demo_nndeploy_dag
```

### Andorid 下运行 demo_nndeploy_dag
```shell
cd PATH/nndeploy/build/install/lib

adb push * /

adb shell 

cd /data/local/tmp/

export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
./demo_nndeploy_dag
```

### 效果示例
```shell
E/nndeploy_default_str: main [File C:\utils\nndeploy\demo\dag\demo.cc][Line 273] start!
digraph serial_graph {
p000000B88A54FB30[shape=box, label=graph_in]
p000000B88A54FB30->p000001C480A1DCF0[label=graph_in]
p000001C480A1DCF0[label=model_0_graph]
p000001C480A1DCF0->p000001C480A51C10[label=model_0_out]
p000001C480A51C10[label=op_link]
p000001C480A51C10->p000001C480A1DAC0[label=op_link_out]
p000001C480A1DAC0[label=model_1_graph]
p000000B88A54FB00[shape=box, label=graph_out]
p000001C480A1DAC0->p000000B88A54FB00[label=graph_out]
}
digraph model_0_graph {
p000000B88A54FB30[shape=box, label=graph_in]
p000000B88A54FB30->p000001C480A52EA0[label=graph_in]
p000001C480A52EA0[label=model_0_graph_preprocess]
p000001C480A52EA0->p000001C480A51310[label=model_0_graph_preprocess_out]
p000001C480A51310[label=model_0_graph_infer]
p000001C480A51310->p000001C480A52FC0[label=model_0_graph_infer_out]
p000001C480A52FC0[label=model_0_graph_postprocess]
p000001C480A5AB50[shape=box, label=model_0_out]
p000001C480A52FC0->p000001C480A5AB50[label=model_0_out]
}
digraph model_1_graph {
p000001C480A5A950[shape=box, label=op_link_out]
p000001C480A5A950->p000001C480A52990[label=op_link_out]
p000001C480A52990[label=model_1_graph_preprocess]
p000001C480A52990->p000001C480A52510[label=model_1_graph_preprocess_out]
p000001C480A52510[label=model_1_graph_infer]
p000001C480A52510->p000001C480A51E50[label=model_1_graph_infer_out]
p000001C480A51E50[label=model_1_graph_postprocess]
p000000B88A54FB00[shape=box, label=graph_out]
p000001C480A51E50->p000000B88A54FB00[label=graph_out]
}
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [model_0_graph_preprocess]!
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [model_0_graph_infer]!
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [model_0_graph_postprocess]!
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [op_link]!
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [model_1_graph_preprocess]!
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [model_1_graph_infer]!
I/nndeploy_default_str: ProcessNode::run [File C:\utils\nndeploy\demo\dag\demo.cc][Line 46] running node = [model_1_graph_postprocess]!
E/nndeploy_default_str: main [File C:\utils\nndeploy\demo\dag\demo.cc][Line 350] end!

C:\utils\nndeploy\build\RelWithDebInfo\demo_nndeploy_dag.exe (进程 10080)已退出，代码为 0。
```


## 基于YOLOV8n的目标的检测

### [下载模型](https://huggingface.co/alwaysssss/nndeploy/blob/main/model_zoo/detect/yolo/yolov8n.onnx)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/blob/main/model_zoo/detect/yolo/yolov8n.onnx
  ```

### [下载测试数据](https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg
  ```

### Windows 下运行 demo_nndeploy_detect
```shell
cd PATH/nndeploy/build/install/bin
.\demo_nndeploy_detect.exe --name NNDEPLOY_YOLOV8 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value C:\huggingface\nndeploy\model_zoo\detect\yolo\yolov8n.onnx --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --input_path C:\huggingface\nndeploy\test_data\detect\sample.jpg --output_path C:\huggingface\nndeploy\temp\sample_output.jpg
```

`注：请将上述PATH更换为自己对应的目录`

### Linux 下运行 YOLOv5s
```shell
cd PATH/nndeploy/build/install/lib
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openVINO 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```

`注：请将上述PATH更换为自己对应的目录`



### Andorid下运行YOLOV5s
```shell
cd PATH/nndeploy/build/install/lib

adb push 

adb push

adb push

export LD_LIBRARY_PATH=PATH/nndeploy/build/install/bin:$LD_LIBRARY_PATH
// mnn 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```

### 效果示例