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
### 编译mnn - mnn mnn-quant mnn-convert
+ ./MNNConvert -f ONNX --modelFile /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx --MNNModel /home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov6m.onnx.mnn --bizCode biz
+ sudo docker run --volume=$(pwd):/workspace -it tnn-convert:latest python3 ./converter.py onnx2tnn /workspace/yolov6m.onnx -optimize -v v3.0 
+ quant目录： /home/always/github/MNN/build/max_release
+ convert目录： /home/always/github/MNN/build/max_release

### windows
+ make install 错误 - setlocal - https://blog.csdn.net/duiwangxiaomi/article/details/124184860?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-124184860-blog-115333145.235%5Ev38%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-124184860-blog-115333145.235%5Ev38%5Epc_relevant_anti_t3_base&utm_relevant_index=1
  + root权限打开vs
  + 修改CMAKE_INSTALL_PREFIX
+ vs无法运行单独运行
+ nndeploy 安装有 lib和bin - 解决
+ 第三库的安装有太多杂项（暂时不管吧）
+ 要不要区分lib和bin呀，第一个是直接解决了windows下的bin和lib问题，tnn的库是区分了的，这里也区分一下吧

### android / ubuntu
+ http://www.taodudu.cc/news/show-864133.html?action=onClick

cmake -DCMAKE_TOOLCHAIN_FILE=/snap/android-ndk-r25c/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_STL="c++_static" -DANDROID_NATIVE_API_LEVEL="android-14" -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true ..
cmake -DCMAKE_TOOLCHAIN_FILE=${NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_STL="c++_static" -DANDROID_NATIVE_API_LEVEL="android-14" -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true ..

+ tensor_rt开address_santizer会crash