# 开发日记

## 2023.08.08
+ 完成根目录的搭建

## 2023.08.10
+ infernce上层架构
+ mnn推理接入的基本结构
+ xxx_inference_param -> 需要去初始化成员变量，我们希望用户尽可能少配置超参数

## 2023.08.12
+ model层架构的开发
+ infer与inference的优化
+ tensorrt的优化

## 2023.08.16
+ git submodule
+ 两外两个仓库
+ 字符串转枚举

## 2023.08.18
+ linux库 - 推理库和模型转换器
  + opencv强烈建议apt-get install
  + TNN、MNN、OpenVINO、ONNXRUNTIME

## 2023.08.19
+ 库卸载 https://blog.csdn.net/get_set/article/details/51276609
### 编译tnn - tnn tnn-quant tnn-convert
+ tnn库按照官方文档安装即可
+ tnn-convert要现安装protobuf
  + protobuf 动态库版本
    + mkdir build
    + cd build
    + cmake .. -DCMAKE_TYPE_SHARED
    + make -j4
    + sudo make install
    + sudo ldconfig（must be）
  + 尝试了很久，还是没有安装成功，故使用tnn官方提供的docker 
    + 参考文档：https://github.com/Tencent/TNN/blob/master/doc/cn/user/convert.md
## 编译mnn - mnn mnn-quant mnn-convert
+ ./MNNConvert -f ONNX --modelFile /home/always/Downloads/TRT2021-MedcareAILab/detr_sim.onnx --MNNModel /home/always/Downloads/TRT2021-MedcareAILab/detr_sim_onnx.mnn --bizCode biz
+ quant目录： /home/always/github/MNN/build/max_release
+ convert目录： /home/always/github/MNN/build/max_release