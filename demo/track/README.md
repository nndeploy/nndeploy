# tracking

## 基于FairMot模型的目标追踪

### 下载模型

- [track/fairmot.onnx](./track/fairmot.onnx): FairMot, Model Type: onnx, input size: Nx640x640x3, classes: 80, [download](https://www.modelscope.cn/models/nndeploy/nndeploy/resolve/master/track/fairmot.onnx)

### 获取测试视频

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4
```

### 运行demo

***`注：请将PATH更换为自己对应的目录`***

- --name: 模型名称
- --inference_type: 推理后端类型
- --device_type: 推理后端的执行设备类型
- --model_type: 模型类型
- --is_path: 模型是否为路径
- --model_value: 模型路径或模型文件
- --codec_flag: 编解码类型
- --parallel_type: 并行类型
- --input_path: 输入视频路径
- --output_path: 输出视频路径
- --model_inputs: 模型输入 
- --model_outputs: 模型输出

#### 推理后端为onnxruntime，推理执行设备为CUDA

```shell
# 进入目录
cd /yourpath/nndeploy/build

# 链接
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH

# 串行执行
./nndeploy_demo_track --name nndeploy::track::fairmot --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --codec_flag kCodecFlagVideo --parallel_type kParallelTypeSequential  --input_path ./person.avi --output_path output.avi --model_value /home/for_all_users/model/track/fairmot/fairmot.onnx --model_inputs im_shape,image,scale_factor --model_outputs fetch_name_0,fetch_name_1

# 耗时
TimeProfiler: demo
------------------------------------------------------------------------------------------
name                            call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
------------------------------------------------------------------------------------------
graph->init()                   1           265.533            265.533            0.000 
graph->run                      1           13108.209          13108.209          0.000 
demo run()                      200         13088.963          65.445             0.000 
decode_node run()               200         749.108            3.746              0.000 
nndeploy::track::fairmot run()  200         10419.823          52.099             0.000 
preprocess run()                200         170.853            0.854              0.000 
infer run()                     200         8823.659           44.118             0.000 
postprocess run()               200         1424.164           7.121              0.000 
vismot_node run()               200         128.972            0.645              0.000 
encode_node run()               200         1789.954           8.950              0.000 
------------------------------------------------------------------------------------------

# 流水线执行
./nndeploy_demo_track --name nndeploy::track::fairmot --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --codec_flag kCodecFlagVideo --parallel_type kParallelTypeSequential  --input_path ./person.avi --output_path output.avi --model_value /home/for_all_users/model/track/fairmot/fairmot.onnx --model_inputs im_shape,image,scale_factor --model_outputs fetch_name_0,fetch_name_1

# 耗时

TimeProfiler: demo
------------------------------------------------------------------------------------------
name                            call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
------------------------------------------------------------------------------------------
graph->init()                   1           230.590            230.590            0.000 
graph->run                      1           9052.425           9052.425           0.000 
decode_node run()               544         8781.201           16.142             0.000 
demo run()                      200         0.017              0.000              0.000 
preprocess run()                200         216.381            1.082              0.000 
nndeploy::track::fairmot run()  200         0.489              0.002              0.000 
infer run()                     200         9031.738           45.159             0.000 
postprocess run()               200         1512.302           7.562              0.000 
vismot_node run()               200         178.980            0.895              0.000 
encode_node run()               200         1966.076           9.830              0.000 
------------------------------------------------------------------------------------------

```

---

## 基于ByteTrack/BotSORT的检测+跟踪

ByteTrack 和 BotSORT 是轻量级的跟踪算法，可与任意检测模型组合使用。

### 架构说明

| 算法 | 特点 | CMake 开关 |
|------|------|-----------|
| **ByteTrack** | 基于 IoU 的简单高效跟踪，不需要重识别特征 | `ENABLE_NNDEPLOY_PLUGIN_TRACK_BYTETRACK` |
| **BotSORT** | 扩展 ByteTrack，加入 ORB 特征匹配的全局运动补偿 (GMC) | `ENABLE_NNDEPLOY_PLUGIN_TRACK_BOTSORT` |

ByteTrack 和 BotSORT 都是 **独立的 DAG 节点**（不是完整图），与检测模型组合使用：

```
VideoDecode → Preprocess → Infer → PostProcess → ByteTrackNode → VisMOT → VideoEncode
                                               ↘ MOTResult ↙
```

### 检测+跟踪 Workflow JSON

```bash
# ByteTrack + YOLO 检测
./nndeploy_demo_track --json_file resources/workflow/track/Track_ByteTrack.json

# BotSORT + YOLO 检测 (带GMC相机运动补偿)
./nndeploy_demo_track --json_file resources/workflow/track/Track_BotSort.json
```

### 程序化 API (C++)

```cpp
#include "nndeploy/track/bytetrack/byte_track_node.h"
#include "nndeploy/track/botsort/bot_sort_node.h"

// 创建 ByteTrack 节点
dag::Node *track_node = graph->createNode<ByteTrackNode>("bytetrack", {detect_edge}, {mot_edge});

// 创建 BotSORT 节点 (需要额外帧输入用于GMC)
dag::Node *track_node = graph->createNode<BotSortNode>("botsort", {frame_edge, detect_edge}, {mot_edge});
```

### 注册节点

ByteTrackNode 和 BotSortNode 已通过 `REGISTER_NODE` 注册，可在 JSON 工作流中按如下 key 引用：

| 节点 | 注册 Key |
|------|---------|
| ByteTrackNode | `nndeploy::track::ByteTrackNode` |
| BotSortNode | `nndeploy::track::BotSortNode` |

---

## 效果示例

#### 输入视频

![sample](../../docs/image/demo/tracking/tracking_sample.jpg) 

#### 输出视频

![result](../../docs/image/demo/tracking/tracking_demo.jpg)