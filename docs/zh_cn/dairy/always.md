
# CANN 学习记录、开发记录、调式记录

## 2024-07-03

### 总结 和 问题
+ 那里寻找华为晟腾的开发资料？
  + [昇腾社区官网-昇腾万里让智能无所不及](https://www.hiascend.com/zh/)
  
    理解其首页，对于开发者而言，最重要的为以下两栏：产品、开发者。

    其中产品栏中 从硬件(边缘推理卡、数据中心推理卡、训练卡等等)->异构计算架构(CANN，编程语言为AscendCL，从开发者视角来看，类似CUDA、CUDA Runtime、cuDNN、cuBlas、TensorRt的集合)->AI框架(接入了torch和tf，原生支持MindSpore，上述都是训练框架，`为什么这里还会有一个推理工具链MindIE呢？`MindIE和AscendCL中推理部分关联是什么呢?)->应用使能->全流程开发工具链

    其中开发者栏中 包含 各类驱动以及软件包的下载，还有最重要的是包含[视频教程](https://www.hiascend.com/edu/courses)

+ 系列硬件（整理一个脑图较为合适）

+ [晟腾AI处理器及CANN软件栈基础](https://www.hiascend.com/edu/courses?activeTab=%E6%A6%82%E8%BF%B0)

  通过学习该堂课程，初步了解了

  + 晟腾 AI core 的基本架构（后续基于文档深入理解）
    + SPMD架构，多个core（`具体是多少个AI core？`），每个core包含
      + 计算单元 - 矩阵运算的cube、向量运算的vector、标量计算的scalar，标量计算单元也充当核内cpu作用，要负责指令接受、调度等等功能（`如何区分fp32、fp16、int8？是不同的数据类型都有一套计算单元吗？`）
      + 内存单元 - global memory + local memory（这个较为复杂，后续需要单独详细理解）+ 寄存器（这个会类似gpgpu中的寄存器文件吗？）
      + DMA 单元 - DMA单元负责数据传输（global memory 与 local memory之间的数据传输）
      + 调度模块（名字是这个吗？负责指令的发送、多个运算单元之间的同步）
  + Ascend C编程语言设计的目的（有五条？后续需再认真回顾）（c++->instricic）
    + 高性能
    + 表达能力
    + ...
  + Ascend C编程语言(初步猜想：像OpenCL和CUDA一样，分为host端和device端，host端负责内存分配、数据传输、调度、控制，device端负责运算)
    + 虽然叫Ascend C，但是在host端就是AscendCL(C++)
    + cube模块的计算原语粒度很大，到了gemm这个层级了
  + 推理
    + AIR（`这个是开源的吗？`最好直接使用或者借鉴该IR，目前框架内部有一套IR的上层数据结构，但最最最最麻烦的还是具体算子的IR，以及各种序列以及反序列化）
    + 模型转换
    + 模型压缩（训练后量化、量化感知训练；剪枝、蒸馏、稀疏化等等主要还是偏算法，这里暂时不关注）
    + 图编译（CANN中图编译比我想象中要复杂很多很多）
  
+ review nndeploy 中 AscendCL Runtime部分代码
  + include/nndeploy/device/ascend_cl
  + src/nndeploy/device/ascend_cl
  
+ review nndeploy 中 AscendCL 推理部分代码
  + include/nndeploy/inference/ascend_cl
  + source/nndeploy/inference/ascend_cl

### 明日安排
+ 挑选 华为昇腾边缘推理盒子 和 开发笔记本
+ CANN文档 - 算子开发部分（尽量解决2024-07-03问题）
+ CANN视频（概述、入门、高级）
+ 目的
  + 硬件体系架构
  + 编程体系架构 与cuda类比
    + 编程粒度

## 2024-07-06
### 总结 和 问题
+ 华为昇腾边缘推理盒子
  + 对于小公司以及个人开发者而言，`开发者套件(Atlas 200I DK A2, 背后的芯片为 加速模块Atlas 200I A2(8TOPS))` 最为方便（开发板卡形式，可以支持摄像头接入、linux桌面），价格合适
  + 购买链接 - https://www.vmall.com/product/10086362347457.html?cid=207641
+ 电脑选购 - 惠普暗夜精灵
+ 达芬奇架构（AI core）
  + 计算单元（scalar（cpu） + vector + cube）（还有很多细节有待展开）
  + 内存单元（global memory + local memory ）（还有很多细节有待展开）
  + 数据搬运单元（DMA）（还有很多细节有待展开）
+ 数据流 + 同步信号流
  + 计算（标量给scalar， 向量给vector，矩阵给cube，异步指令流）
  + 计算会产生依赖，数据依赖（需要同步信号流）、控制依赖（scalar cpu解决吧）
+ Ascend C 
  + 以ADD入门
    + host api
    + device api(多级API)
  + 优化方法
    + double buffer
+ 2024-07-03遗留问题
  + CANN已经包含了推理工具链（AscendCL-aclmdl），为什么这里还会有一个推理工具链MindIE呢？
    + 官方简介：昇腾推理引擎，基于昇腾硬件的运行加速、调试调优、快速迁移部署的高性能深度学习推理框架，分层开放满足各类需求，统一接口使能极简开发，沉淀能力构筑极致性能
    + 个人理解（较为片面）：基于CANN（AscendCL-aclmdl）
      + 开发MindRT(接口会有点点类似Trt，提高易用性，可以通过onnx直接创建推理示例、也可以手动构图)
      + 针对大模型场景，往往通用的推理接口不好用，所以有了（MindIE-SD和MindIE-LLM，类似英伟达的FasterTransformer）
      + MindIE-Service: 推理服务化框架（类似英伟达的Inference Triton Server）
  + 具体是多少个AI core？
    + 每个芯片都包含多个AI core（每个AI core只共享主存），具体依据芯片型号不同而不同（我没有找到具体的资料）
  + 计算单元 - 矩阵运算的cube、向量运算的vector、标量计算的scalar，标量计算单元也充当核内cpu作用，要负责指令接受、调度等等功能（`如何区分fp32、fp16、int8？是不同的数据类型都有一套计算单元吗？`）
    + 由矩阵运算的cube、向量运算的vector、标量计算的scalar内部去区分不同的计算类型，以矩阵运算的cube单元（Atlas 200I A2 的cube）为例
      + 内部的一个cube既可以做fp16，也可以做int8，fp16(16x16的fp16矩阵计算、32x16或者16x32的int8矩阵计算)
  + AIR（`这个是开源的吗？`最好直接使用或者借鉴该IR，目前框架内部有一套IR的上层数据结构，但最最最最麻烦的还是具体算子的IR，以及各种序列以及反序列化）
    + 需要进一步查找资料，我想用这个IR作为nndeploy里面的IR

### 明日安排
+ AIR
  + 看能不能将nndeploy的IR和AIR进行整合，或者直接使用AIR
+ 完善nndeploy中目前已接入的CANN（aclrt[runtime]和aclmdl[推理]）


## 2024-07-07
### 总结 和 问题
+ alcrt的接口
  + alcrt初始化
  + 使能设备
  + 创建上下文
  + 创建流
  + 管理多流
  + 同步
  + 内存分配
  + 内存拷贝
  + 销毁流
  + 销毁上下文
  + 重置设备
  + alcrt反初始化
+ 将aclrt与nndeploy-device模块结合：设备是nndeploy对硬件设备的抽象，通过对硬件设备的抽象，从而屏蔽不同硬件设备编程模型带来的差异性，初步完成对AscendCL设备的接入。主要功能如下
  + **统一的内存分配**：为不同设备提供统一的内存分配接口，从而可简化数据容器`Buffer`、`Tensor`的内存分配
  + **统一的内存拷贝**：为不同设备提供统一的内存拷贝接口（设备间拷贝、主从设备间上传/下载），从而可简化数据容器`Buffer`、`Tensor`的内存拷贝
  + **统一的同步操作**：为不同设备提供统一的同步操作接口，可简化设备端模型推理、算子等同步操作
  + **统一的硬件设备信息查询**：为不同设备提供统一的硬件设备信息查询接口，帮助用户更好的选择模型全流程部署的运行设备
+ aclrt的多流有优化空间，可以让模型推理或者算子加速吗？

### 明日安排
+ 推理框架构图
  + onnx构图
  + pytorch构图
  + tensorrt api
  + ggml 构图
  + aclge 构图
  + mnn expr构图

## 2024.07.20
+ aclnn - 算子接口（这个肯定可以实现，构图依赖nndeploy本身）
  + dataformat
    + 对外
    + 对内
+ atb - Ascend Transformer Boost(主要是服务语言模型提供的算子库)
+ ge - 手动构建图，权重如何加载进来呢？那为什么还要做这件事情呢？

### 综合来看，针对yolo这类小模型，优先使考虑aclnn，后续要做类似llama.cpp时，再来考虑atb，直接pass ge

## 2024.07.21
+ 写一下头文件，逻辑清晰
+ 两段接口，有没有性能损失呢？
+ 每个算子对外和对内的data_format到底是怎样的呢？
+ 还有那些帮助函数要写呢？
  + 各种convert
+ 单算子测试，等板卡回来就开始测试
+ 内存优化的策略可行吗？
+ 针对动态shape好搞吗
+ git submodule add git@github.com:onnx/onnx.git third_party/onnx
+ git submodule add git@github.com:protocolbuffers/protobuf.git third_party/protobuf

CMake Error in third_party/onnx/CMakeLists.txt:
  export called with target "onnx_proto" which requires target "libprotobuf"
  that is not in any export set.


## 2024.07.27
+ 从ir开始梳理 - 与onnx对齐
+ 稍微梳理一下op
+ 梳理解释器 - 完善onnx解释器
+ 增加上层op


## 2024.07.27
+ 完善上层op
+ 增加华为晟腾算子
+ 搭建cann开发环境

Released Versions
ONNX version 	IR version 	Opset version ai.onnx 	Opset version ai.onnx.ml 	Opset version ai.onnx.training
1.0 	3 	1 	1 	-
1.1 	3 	5 	1 	-
1.1.2 	3 	6 	1 	-
1.2 	3 	7 	1 	-
1.3 	3 	8 	1 	-
1.4.1 	4 	9 	1 	-
1.5.0 	5 	10 	1 	-
1.6.0 	6 	11 	2 	-
1.7.0 	7 	12 	2 	1
1.8.0 	7 	13 	2 	1
1.8.1 	7 	13 	2 	1
1.9.0 	7 	14 	2 	1
1.10.0 	8 	15 	2 	1
1.10.1 	8 	15 	2 	1
1.10.2 	8 	15 	2 	1
1.11.0 	8 	16 	3 	1
1.12.0 	8 	17 	3 	1
1.13.0 	8 	18 	3 	1
1.13.1 	8 	18 	3 	1
1.14.0 	9 	19 	3 	1
1.14.1 	9 	19 	3 	1
1.15.0 	9 	20 	4 	1
1.16.0 	10 	21 	5 	1


Finish! Here is the difference:
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Add        │ 8              │ 8                │
│ Concat     │ 19             │ 19               │
│ Constant   │ 139            │ 139              │
│ Conv       │ 64             │ 64               │
│ Div        │ 1              │ 1                │
│ MaxPool    │ 3              │ 3                │
│ Mul        │ 58             │ 58               │
│ Reshape    │ 5              │ 5                │
│ Resize     │ 2              │ 2                │
│ Sigmoid    │ 58             │ 58               │
│ Slice      │ 2              │ 2                │
│ Softmax    │ 1              │ 1                │
│ Split      │ 9              │ 9                │
│ Sub        │ 2              │ 2                │
│ Transpose  │ 2              │ 2                │
│ Model Size │ 12.2MiB        │ 12.2MiB          │
└────────────┴────────────────┴──────────────────┘

## 2024.08.24
+ 初步解决的编译和net以及session的运行时问题
+ 但是单算子调用出现了错误，错误为内部错误，但是模型又可以运行
  + 原因：我买的是华为昇腾310b的推理卡，官网显示不能单算子模式调用，那我这种形式该怎么做呢，去910上搞吗


## 2024.09.05

+ 更换git submodule

  git submodule add https://github.com/mlc-ai/tokenizers-cpp.git third_party/tokenizers-cpp


## 2024.09.17
+ atc --model=./yolov8n.onnx --output=./yolov8n.onnx.om --framework=5 --soc_version=Ascend310B


## 2024.10.05
+ 完成目录的修改
  + interpret -> ir
  + ir -> ir


  [submodule "python/pybind11"]
	path = python/pybind11
	url = https://github.com/pybind/pybind11.git

  git submodule add https://github.com/pybind/pybind11.git third_party/pybind11


## 问题
+ split算子在有些昇腾910上会运行失败(910b4不会)
+ transpose参数写错了
+ 8.0之后的310b的卷积算子运行出错
+ CREATE_EXECUTOR运行出错
  + 很多头文件出错 


## atd

+ atc --model=/home/ma-user/work/github/nndeploy/build/yolov8n_debug.onnx --framework=5 --output=/home/ma-user/work/github/nndeploy/build/yolov8n_debug.onnx.om --soc_version=Ascend910B4

+ atc --model=/home/ma-user/work/github/nndeploy/build/modified_yolov8n.onnx --framework=5 --output=/home/ma-user/work/github/nndeploy/build/modified_yolov8n.onnx.om --soc_version=Ascend910B4

