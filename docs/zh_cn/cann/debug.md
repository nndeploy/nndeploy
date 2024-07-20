
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
    + SPMD架构，多个core（`具体是多少个core？`），每个core包含
      + 计算单元 - 矩阵运算的cube、向量运算的vector、标量计算的scalar，标量计算单元也充当核内cpu作用，要负责指令接受、调度等等功能（如何区分fp32、fp16、int8？`是不同的数据类型都有一套计算单元吗？`）
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