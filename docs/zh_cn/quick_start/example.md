# 示例工程

## 从源码编译

参考编译文档，[编译文档链接](./build.md)

## 基于DAG的模型部署演示示例（采用默认config.cmake即可编译成功）

### Linux 下运行 demo_nndeploy_dag
```shell
cd PATH/nndeploy/build/install/lib
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
./demo_nndeploy_dag
```

### Windows 下运行 demo_nndeploy_dag
```shell
cd PATH/nndeploy/build/install/bin
.\demo_nndeploy_dag.exe
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
E/nndeploy_default_str: main [File C:\utils\nndeploy\demo\dag\demo.cc][Line 271] start!
digraph serial_graph {
p00000093FA1AF720[shape=box, label=graph_in]
p00000093FA1AF720->p000001CB69CFAD70[label=graph_in]
p000001CB69CFAD70[label=model_0_graph]
p000001CB69CFAD70->p000001CB69D25D20[label=model_0_out]
p000001CB69D25D20[label=op_link]
p000001CB69D25D20->p000001CB69CEBA80[label=op_link_out]
p000001CB69CEBA80[label=model_1_graph]
p00000093FA1AF6F0[shape=box, label=graph_out]
p000001CB69CEBA80->p00000093FA1AF6F0[label=graph_out]
}
digraph model_0_graph {
p00000093FA1AF720[shape=box, label=graph_in]
p00000093FA1AF720->p000001CB69D26CE0[label=graph_in]
p000001CB69D26CE0[label=model_0_graph_preprocess]
p000001CB69D26CE0->p000001CB69D26080[label=model_0_graph_preprocess_out]
p000001CB69D26080[label=model_0_graph_infer]
p000001CB69D26080->p000001CB69D267D0[label=model_0_graph_infer_out]
p000001CB69D267D0[label=model_0_graph_postprocess]
p000001CB69D2B7E0[shape=box, label=model_0_out]
p000001CB69D267D0->p000001CB69D2B7E0[label=model_0_out]
}
digraph model_1_graph {
p000001CB69D2B360[shape=box, label=op_link_out]
p000001CB69D2B360->p000001CB69D273A0[label=op_link_out]
p000001CB69D273A0[label=model_1_graph_preprocess]
p000001CB69D273A0->p000001CB69D25C00[label=model_1_graph_preprocess_out]
p000001CB69D25C00[label=model_1_graph_infer]
p000001CB69D25C00->p000001CB69D25FF0[label=model_1_graph_infer_out]
p000001CB69D25FF0[label=model_1_graph_postprocess]
p00000093FA1AF6F0[shape=box, label=graph_out]
p000001CB69D25FF0->p00000093FA1AF6F0[label=graph_out]
}
E/nndeploy_default_str: main [File C:\utils\nndeploy\demo\dag\demo.cc][Line 348] end!

C:\utils\nndeploy\build\RelWithDebInfo\demo_nndeploy_dag.exe (进程 22748)已退出，代码为 0。
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

### Windows下运行YOLOv5s
```shell
cd PATH/nndeploy/build/install/bin
export LD_LIBRARY_PATH=PATH/nndeploy/build/install/bin:$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openvino 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
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