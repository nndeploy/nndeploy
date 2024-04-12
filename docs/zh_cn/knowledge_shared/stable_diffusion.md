
# stable diffusion

## 1 目标

1.1. 基于DAG部署stable diffusion系列的模型，形成类似[comfyui](https://github.com/comfyanonymous/ComfyUI)的产品。（可以形成产品，直接给CV AIGC的用户用）

1.2. 置底向上理解并实现stable diffusion(算子->框架->算法)，形成类似[ggml](https://github.com/ggerganov/ggml)、[llama.cpp](https://github.com/ggerganov/llama.cpp)、[stable_diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)的产品

## 2 why

主要一下三个层面

### 2.1 学习层面

2.1.1. 从推理部署的角度出发，理解stable diffusion系列模型

2.1.2. 基于stable diffusion，从0到1理解并实现推理子模块

2.1.3. 高性能的op开发（cuda为主）

### 2.2 科研层面

从算法、部署、推理角度理解并实现stable diffusion，可以做出一些科研价值的产出，例如

2.2.1. 高性能的算子 - 类似flash attension算子、基于triton开发
   
2.2.2. 推理子模块 - 高效的内存管理、图优化等

2.2.3. stable diffusion的模型量化

### 2.3 产品层面

2.3.1. 为用户提供类似[comfyui](https://github.com/comfyanonymous/ComfyUI)的产品

2.3.2. 对于stable diffusion模型部署而言，提供另外一个选择


**总结**：AI迅速发展，通过这个开源框架置底向上理解并实现stable diffusion(算子->框架->算法)。或许在下一个AI风口到来的时候，我们可以在AI infra这个方向上，能够跟得上节奏。

## 3 how

### 3.1 算法侧

#### 3.1.1 调研阶段

1. 经典stable diffusion模型的详细分析
   1. 算法原理 - 达到及格水平
   2. 训练原理（数据集、loss）- 达到及格水品
   3. 模型结构的详细分析 - 达到良好的水品
      1. 有几个模型
      2. 模型如何串联起来
      3. 模型的前后处理
      4. 模型的前后处理、推理的参数
      5. 每个模型的结构

#### 3.1.2 开发阶段(可选项)

1. 基于主流的框架，跑通stable diffusion的推理
   1. pytorch
   2. hugging face
   3. stable_diffusion.cpp
   4. stable fast
   5. confyui
   6. sd webui
   7. ...

#### 3.1.3 进阶阶段(可选项)

1. 可以大概知道主流框架实现的基本原理

### 3.2 推理子模块 （这一部分我比较熟悉一点，故写的多一些）

通过我们对算法原理理解，以一种垂直的方式把stable diffusion给做出来。完整的推理框架包含以下子模块，我们会省略大部分模块，关注我们认为最不可或缺的部分 - 通过手动搭建计算图的方式实现模型推理（综合ggml+stable_diffusion.cpp+tensorrt api+mnn+tnn）

- 模型转换（将训练框架模型文件转化为权重文件）
  - 简介：将训练框架模型文件转换推理框架自定义模型文件
  - 输入：训练框架模型文件
  - 输出：自定义模型文件
- 模型解释（可选项）
  - 简介：读入模型文件，并将模型文件解释为中间表示
  - 输入: 自定义模型文件
  - 输出：中间表示IR (**要做**)
- 手动搭建计算图（**要做**）
  - 简介：类似ggml与tensorrt api，通过手动搭建计算图
  - 输入: 
    - 对模型的网络结构的理解
    - 权重文件
  - 输出
    - 计算图
- 模型推理初始化(**要做**)
  - 简介：为模型推理做好一切准备 
    - 整个推理框架最核心最核心的部分，模型推理依赖该部分，模型离线量化也依赖该部分
  - 输入: 
    - 计算图 或者 中间表示IR（Option） 
    - 推理的超参数
  - 输出
    - 一个可执行推理的计算图
- 模型推理(**要做**)
  - 简介：写入输入，得到输出
  - 输入：输入的tensor
  - 输出：推理后的tensor
- 模型训练后量化（可选项）
  - 简介：量化模型文件，使得模型文件更小，期望模型可以跑的更快
  - 输入：fp32的模型文件
  - 输出：int8的模型文件
- 模型训练（暂时不做，可选项）
  - 简介：可实现模型训练
  - 输入：
    - 通过算子表达式搭建模型文件
    - 超参数
  - 输出：可在线训练的模型

#### 3.2.1 调研阶段

1. 理解推理框架的实现
   1. ggml+stable_diffusion.cpp（初步理解）
   2. tensorrt api（初步理解）
   3. mnn（初步理解）
   4. tnn（初步理解）
   5. stable fast（可选项）
   6. ...

#### 3.2.2 开发阶段

1. 基础的数据结构 - data_type\data_format\pricision等等（nndeploy已完成）
2. 数据容器 - 已有一个初版的tensor的定义，下一步计划
   1. 合并mat和tensor，重新设计
      1. 之前考虑分开的原因是：
        1. Tensor主要是服务模型的输入输出、中间变量。就是他会注重shape、data_format的方面，会注重扩展性（比如服务与量化的Tensor、分布式上Tensor等，基于模型的内存池分配等）；处于Tensor的特性基本不用考虑浅拷贝，深拷贝等
        2. Mat主要是为了op后续替代opencv提出的结构呀，会注重操作简便性，比如浅拷贝、深拷贝等，也没有data_format的要求。
      2. 现在想要合并原因为：
         1. cv类的算子可以加入 到 推理子模块 里面一起构图了；
         2. 利用推理子模块，cv类的算子本身也可以自己构图
         3. 稍微稍微了解了下cvcuda，它里面的数据结构就是Tensor
   2. 考虑tensor的protobuf的序列化（可选项）
   3. 考虑流水线并行（可选项）
   4. 考虑分布式（可选项）
3. ir的设计
   1. 参考onnx已经初步实现一个版本
   2. 参考MindIr再次优化
   3. 考虑多个权重模型文件（可选项）
   4. 考虑protobuf的序列化
4. 手动构图
   1. 接口的形式
      1. op
      2. forward
   2. 考虑手动构图（基于对推理的理解，参考已有的dag实现）
   3. 考虑之后从ir自动构图（可选项）
5. 模型推理初始化
   1. 委托给executor
   2. 内存预分配
   3. 图优化（可选项）

#### 3.2.3 进阶阶段(可选项)

1. 分布式
2. 自动构图
  

### 3.3 高性能算子

#### 支持的平台
- cuda
- opencl
- cpu

#### 四个需求
- 自动构图（封装即可）
- 手动构图（封装即可）
- 可以导出为类似blas或者opencv的接口调用形式（封装即可）

#### 实现方式 
- 调库
  - cudnn
  - oneAPI
  - qnnpack
  - ...
- 基于triton实现
- 手写

#### 算子的测试方式





