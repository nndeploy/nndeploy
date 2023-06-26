
# hugging face模型工程化


## 环境
+ pytorch
+ tensorflow
+ transform
+ onnx
+ onnxruntime
+ onnxsimplifer
+ onnxoptimizer
+ tensor_rt
+ cudnn
+ cublas
+ cuda


## 模型以及推理代码
+ 去 hugging face下载模型文件
+ 去 huging face找到模型文件对应的github地址
  + 直接包含推理代码
  + 不包含推理代码
    + 看训练代码，看dataset的前处理
    + 看test的后处理
+ 去 hugging face 


## 模型转换
+ 把hugging face上的bin后缀的模型文件转换为对应训练框架的模型文件（例如pytorch的pt模型文件）
+ 两种方式
  + 把训练框架的模型文件直接转换为推理框架所需的模型文件
  + 把训练框架的模型文件转换为onnx模型文件
    + 使用onnxsim以及onnxopt对模型进新图优化
    + 直接使用该onnx模型文件或者把onnx模型文件转换为推理框架的模型文件格式


## c++部署
+ 前处理以及前处理参数
+ 后处理以及后处理参数
+ 编写demo