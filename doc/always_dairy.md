# 要做的事项
+ 建立起基本的共识
+ cmake结构
+ 第三库的管理
+ clang-tidy、clang-format 统一编码风格
+ doc的调研 注释的风格
+ base模块 c++设计与实现 分工合作
+ device 
  + cpu模块、X86M模块、arm模块、opencl模块、cuda模块
  + buffer
  + mat
+ inference
  + tf-lite
  + tensor_rt
  + openvino
  + coreml
+ graph
  + 思路，设计
+ vision
  + cv算子


# 23
+ DataType 的 type_trait
+ basic以及Desc的构造函数
  + 强转问题
+ buffer的多线程资源管理问题
+ tensor
  + 要不要增加不存在buffer，但是其他都有的接口
+ 移除include_c_cc.h


# 26
+ 多线程
+ c++语法
+ 标准库
+ "[\s\S]*" 

+ export LD_LIBRARY_PATH=/home/always/github/nndeploy/build/install:$LD_LIBRARY_PATH    
+ export LD_LIBRARY_PATH=/home/always/github/nndeploy/third_party/user/linux/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH
+ export PATH=/snap/bin/:$PATH
+ export LD_LIBRARY_PATH=/home/always/github/nndeploy/build/install/:$LD_LIBRARY_PATH
+ export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH