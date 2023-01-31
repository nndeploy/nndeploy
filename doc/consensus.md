# 开发共识

## 成为AI算法落地领域最核心的部署框架

## 语言
+ c++
+ python
+ hpc
  + cuda
  + opencl -- 支持OpenGL数据零拷贝到OpenCL
  + metal
  + neon
  + x86
+ camke
+ 尽量不采用bat以及shell等脚本语言

## 序列化
+ json
+ protobuf
+ csv

## 写代码之前必做
+ 初步设计
+ 阅读书籍
+ 阅读并借鉴开源代码，借鉴了任何开源代码都必须在reference.md中注明
+ 精细设计
+ 代码

## 重要性
+ 不着急
+ 设计
  + 架构
  + 模块
  + class
+ 功能完备（fastdeploy不支持动态shape、openvino不支持外部传入内存）
+ 模块化
  + 内部开发效率
+ 扩展性
  + 针对硬件公司
+ 易用性
  + 外部开发效率
+ 性能
  + 时间
  + 空间

## 机器学习部署框架 -> 需要达到一下目的，需要大量机器学习系统、hpc、ai算法、编译、后台
+ AI算法工程化工程师
  + 开发效率
  + 性能
    + c++
    + 部署
    + 模型
+ 算法工程师
  + 基于python接口框架开发
  + 一键切换推理框架，代替onnxruntime
  + 对齐
+ 算法落地的公司
  + 算法落地
    + 开发效率
    + 性能
+ 芯片公司
  + 帮助其对接算法公司
  + 统一的接口
+ 真正统一的nn部署框架 -- python -> so(tvm and ai template)

## 训练框架
+ pytorch
+ onnx


