# ocr

## PaddleOCR

### 下载模型

- [ocr/OCRv5_mobile_det/inference.onnx](./ocr/OCRv5_mobile_det/inference.onnx): OCRv5_mobile_det, Model Type: onnx, [download](https://modelscope.cn/models/nndeploy/nndeploy/resolve/master/ocr/OCRv5_mobile_det/inference.onnx)

- [ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.onnx](./ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.onnx): ch_ppocr_mobile_v2.0_cls_infer, Model Type: onnx, [download](https://modelscope.cn/models/nndeploy/nndeploy/resolve/master/ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.onnx)

- [ocr/OCRv5_mobile_rec/inference.onnx](./ocr/OCRv5_mobile_rec/inference.onnx): OCRv5_mobile_rec, Model Type: onnx, [download](https://modelscope.cn/models/nndeploy/nndeploy/resolve/master/ocr/OCRv5_mobile_rec/inference.onnx)

- [ocr/OCRv5_mobile_rec/config.json](./ocr/OCRv5_mobile_rec/config.json): OCRv5_mobile_rec, Type: json, [download](https://modelscope.cn/models/nndeploy/nndeploy/resolve/master/ocr/OCRv5_mobile_rec/config.json)


### 下载图片

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

### 运行demo

***`注：请将PATH更换为自己对应的目录`***

#### 运行flag介绍

- --name: 模型名称
- --inference_type: 推理后端类型
- --device_type: 推理后端的执行设备类型
- --model_type: 模型类型
- --is_path: 模型是否为路径
- --detector_model_value: 检测模型文件路径
- --classifier_model_value: 方向分类模型文件路径
- --recognizer_model_value: 识别模型文件路径
- --character_txt_value: 识别模型配置文件路径

- --codec_flag: 编解码类型
- --parallel_type: 并行类型
- --input_path: 输入图片路径
- --output_path: 输出图片路径
- --classifier_model_inputs: 方向分类模型输入
- --recognizer_model_inputs: 识别模型输入
- --detector_model_inputs: 检测模型输入
- --classifier_model_outputs: 方向分类模型输出
- --recognizer_model_outputs: 识别模型输出
- --detector_model_outputs: 检测模型输出

```shell
# 进入目录
cd /yourpath/nndeploy/build

~/.cache/modelscope/hub/models/chunquansang/paddleocr_detector_onnx/
# 执行
./nndeploy_demo_ocr --name nndeploy::ocr::DetectorGraph --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --classifier_model_value ~/.cache/modelscope/hub/models/nndeploy/nndeploy/ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.onnx --detector_model_value ~/.cache/modelscope/hub/models/nndeploy/nndeploy/ocr/OCRv5_mobile_det/inference.onnx --recognizer_model_value ~/.cache/modelscope/hub/models/nndeploy/nndeploy/ocr/OCRv5_mobile_rev/inference.onnx --character_txt_value ~/.cache/modelscope/hub/models/nndeploy/nndeploy/ocr/OCRv5_mobile_rec/config.json --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --detector_model_inputs x --detector_model_outputs fetch_name_0 --classifier_model_inputs x --classifier_model_outputs softmax_0.tmp_0 --recognizer_model_inputs x --recognizer_model_outputs fetch_name_0 --input_path ./12.jpg --output_path ./output.jpg

# 耗时
----------------------------------------------------------------------------------------------
name                                call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
----------------------------------------------------------------------------------------------
graph->init()                       1           155.885            155.885            0.000 
demo init()                         1           155.860            155.860            0.000 
decode_node init()                  1           0.010              0.010              0.000 
infer init()                        1           38.942             38.942             0.000 
c_infer init()                      1           23.478             23.478             0.000 
r_infer init()                      1           93.239             93.239             0.000 
encode_node init()                  1           0.003              0.003              0.000 
graph->run                          1           5528.289           5528.289           0.000 
demo run()                          10          5528.214           552.821            0.000 
decode_node run()                   10          97.330             9.733              0.000 
nndeploy::ocr::DetectorGraph run()  10          1937.117           193.712            0.000 
preprocess run()                    10          82.791             8.279              0.000 
infer run()                         10          1831.878           183.188            0.000 
postprocess run()                   10          22.311             2.231              0.000 
RotateCropImage run()               10          27.796             2.780              0.000 
Classifier run()                    10          142.685            14.268             0.000 
c_preprocess run()                  10          3.425              0.343              0.000 
c_infer run()                       10          139.091            13.909             0.000 
c_postprocess run()                 10          0.075              0.008              0.000 
RotateImage180 run()                10          0.027              0.003              0.000 
Recognizer run()                    10          3243.843           324.384            0.000 
r_preprocess run()                  10          12.431             1.243              0.000 
r_infer run()                       10          2587.646           258.765            0.000 
r_postprocess run()                 10          643.618            64.362             0.000 
PrintOcrNode run()                  10          3.585              0.359              0.000 
DrawDetectorBox run()               10          6.029              0.603              0.000 
encode_node run()                   10          69.521             6.952              0.000 
----------------------------------------------------------------------------------------------

