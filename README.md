# 机器学习部署框架

# 目录
+ build
  + 编译后工程路径
+ cmake
  + 通用的cmake，例如cpack, install
  + config.cmake，编译模板config
  + 第三方库的cmake，目录的形式 + 链接进工程
+ doc
  + 文档，需要调研文档工具以及写注释的方式
+ docker
  + docker脚本
+ ffi
  + 多语言支持
+ nnconvert
  + 统一的模型转换器，完成 X模型文件 -> Y模型文件 功能
  + 配合使用的推理框架版本共同使用
  + 在docker里面统一管理
+ nndeploy -> 核心 的 c++部署框架
  + base 基础模块以及帮助函数，只依赖标准库
  + device 设备管理模块，非常核心且重要的模块，依赖base以及对应设备第三方库
    + 架构类
    + 运行设备类
    + buffer类
    + mat类
    + 内存管理类
  + cryption 加解密模块，依赖base，依赖特殊的加解密第三方库，门面模式 + 适配器模式 + 单例模式
  + audio 音频算子，依赖base,device,cryption
  + vision 视频算子，依赖base,device,cryption
  + aicompile ai编译器，依赖base,device,cryption
  + inference 推理款该，依赖base,device,cryption
  + graph 图管理模块，非常核心且重要的模块，具体算法基于该模块建立，
    + 依赖base,device,cryption
    + 通过编译宏依赖如下平行四个模块 
      + audio
      + vision
      + aicompiler
      + inference
    + 具体子子模块
      + config, 用于传参
      + packet, 数据传输，边
      + packet_manager, 数据管理模块
      + node, 执行节点
      + node_manager, 节点管理模块
      + graph, 执行图
+ nnedit
  + 基于onnx或者pytorch的模型编辑器
+ nnlightweight
  + 模型轻量化工具
+ nnoptimize
  + 基于onnx或者pytorch模型图优化工具
+ nnquantize
  + 基于推理框架简历的抽象的模型量化工具，在docker里面统一管理
  + 基于onnx和pythorch模型量化工具
+ nntask
  + 各类落地的ai算法
+ third_party
  + git - 第三方库源码
  + release - 第三方库