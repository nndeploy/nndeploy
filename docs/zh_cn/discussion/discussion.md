# 开发者讨论

## mac
+ 不开启任何推理框架编译通过
+ 完善coreml
  + 代码
  + 编译
  + 调通yolov5
+ 调通mnn
  + 编译
  + 调通yolov5

## nndeploy 的 github Actions
+ nndeploy采用github官网的Actions做编译的ci测试，出现了如下几个问题
  + mac下编译不通过，这个可能是mac下编译有问题
  + 无法每次push都进行一次编译测试

## mdc
+ 华为的那些芯片可以使用设备mdc和推理mdc呀
+ 设备Mdc
  + 叫mdc是否合理呢？是不是要叫它本身的编程模型Ascend或者acl呢？（类似编程模型cuda、opencl）（这个host端的api跟cuda十分类似，这个是不是就是一种cuda兼容呀，它得设备端得代码和cuda类似吗）
  + 是否可以在其他芯片上使用吗？（例如昇腾系列的芯片上）
+ 推理框架Mdc
  + 叫mdc是否合理呢？与 CANN 是什么关系呀
  + 是否可以在其他芯片上使用吗？（例如昇腾系列的芯片上）
+ 这个与mindspore-lite是什么关系呀
+ 变量名是 全部小写呀？
+ OrtValueInfo这个是不是改一下比较好呀
+ 在readme中需要备注以下，主要是那个操作系统端运行，可以在那些芯片上上跑起来，是否要把它添加到主页的README上呢
+ aclrt -> cuda runtime api
+ aclmdl -> 这个mdl是指什么含义呀

### v2修改
+ 部分头文件中使用的是_MDC_和mdc
+ 变量名全部小写
+ 代码中ASCENDCL以及AscendCL的字段改为 -> AscendCL，有如下几个原因
  + AscendCL是官方写法
  + 内部的写法比较倾向AscendCL， 例如设备中OpenCL、推理框架中的TensorRt
+ 设备类Device增加void* getContext()接口
  + 对于设备而言，通常都有上下文环境以及执行队列的概念
  + 在inference/ascend_cl中有获取上下文的需求
+ 在run函数里面，这里每次都重新创建数据集，却没有去销毁，这里有问题吗？
```c++
base::Status AscendCLInference::run() {
  base::Status status = base::kStatusCodeOk;

  // Always: 
  input_dataset_ = aclmdlCreateDataset();
  output_dataset_ = aclmdlCreateDataset();
}
```

## 流水线并行

流水线并行实现细节讨论

### nndeploy中的流水线并行是什么

举例说明。某台Linux服务器上， 有n张4090显卡。

相册头像聚类任务：（原始图片Edge）-> 人脸检测Node -> (人脸Edge) -> 人脸关键点Node —> (人脸关键点Edge) -> 人脸比对Node -> (人脸id Edge)

注：假设相册中有1000张图片

进一步说明：

+ 1. 分配三个线程，每个线程分别处理 人脸检测（Node）、人脸关键点（Node）、人脸比对（Node）
+ 2. （原始图片） 有个数据仓库 （Edge），当该数据仓库有数据时，人脸检测可以开始处理
+ 3. （人脸） 有个 数据仓库 （Edge），当该数据仓库有数据时，人脸关键点可以开始处理
+ 4. （人脸关键点） 有个 数据仓库 （Edge），当该数据仓库有数据时，人脸比对可以开始处理
+ 5. (人脸id) 有个 数据仓库 （Edge），当该数据仓库有数据时，调用方可得到数据


### 初步解决方案

+ 开一个三个线程的线程池
+ 将三个节点都放到线程池中
+ 每个线程执行都要分别都要等待（原始图片Edge） (人脸Edge) (人脸关键点Edge) 有数据且数据已经写入成功才能执行
+ 调用方线程(人脸id Edge)有数据且数据已经写入成功

### 具体疑难问题
+ 对于用户而言，流水线并行 和 串行 是一样的执行模式
+ 在执行graph->run的时候做到，只要一次将三个节点放到线程池
+ 主线程如何知道是最后一张图片，然后不在哪里等待呢
+ 开始节点和中间节点对数据的要求不一致，开始节点输入Edge有数据即可，中间节点输入Edge有数据且该数据已完成写入（即上一个节点已经完成写入输出Edge）。（原始图片） 有个数据仓库 （Edge）是当该数据仓库有数据时，第一个任务就可以运行
+ 如何做到数据销毁

## 文档的结构讨论
### 要达到一下几个目标
+ 适合readthedocs的排版
+ 区分中文文档和英文文档
+ 要能够兼容部署类文章的分享（我比较愿意把文章写在这个这个文件夹下面）
+ TODO - API自动导出

### 文档的结构
+ [tvm的格式](https://tvm.apache.org/docs/) - 这个不是readthedocs的格式
+ [mnn的格式](https://mnn-docs.readthedocs.io/en/latest/index.html)
+ [mmdeploy的格式](https://mmdeploy.readthedocs.io/zh-cn/latest/get_started.html)

### 基于守夜大佬目录修改
+ 开始
  + readme（想把这些都写在readme里面，主要是想要提升开发效率，不想多次反复写同样的文章）
    + 发展与展望（这些算readme吗）
    + 当前支持的硬件、推理框架（这些算readme吗）
    + 当前已经跑通的demo算法（这些算readme吗）
    + 当前已有的特性（这些算readme吗）
    + 近期要做的事情（roadmap）（这些算readme吗）
    + 最终的愿景（这些算readme吗）
  + 编译和安装nndeploy
  + 如何跑通nndeploy
  + 贡献指南
    + pr规则、代码规范、微信群、可参与贡献的角度等
+ 用户指南
  + 部署一个新算法
    + 导出模型
    + 构建pipeline
    + 前处理、后处理
  + 增加一个新推理框架（这个算用户指南吗）
  + 增加一个新设备（这个算用户指南吗）
  + 增加一个新op（这个算用户指南吗）
+ 开发者指南
  + 整体架构
  + 几大组件的作用和依赖关系 对应到文件夹
  + 数据容器
    + Tensor、Mat、Buffer
  + 有向无环图
    + graph、node、edge
  + 资源池
    + threadpool设计
    + memorypool设计
  + 并行方式
    + 流水线并行
    + 任务并行
+ 知识分享
  + 模型导出
  + 量化
+ 问题讨论

## dag 代码review
+ condition和loop是Graph而不是Node
  + Graph继承Node
  + 相比Node，Graph区别主要如下
    + Graph具备管理Node的能力
    + Graph的执行委托executor
+ 把executor.h挪到主目录下来
+ 并把executor.h中的帮助函数放到util中来
+ 命名的修改
  + condition_is_running -> runnint_condition
  + graph目录 -> executor目录
  + condition_parallel_pipeline_executor -> parallel_pipeline_condition_executor
+ 解决Graph嵌入Graph偶发性错误问题
+ 功能验证
  + 一个图中各个子模块都有各自的并行方式（串行、任务并行、流水线并行）
  + 当一个Edge既是整个图的输出也是某个中间节点的输入，这个该如何解决（主要麻烦点在流水线并行模式，待review解决）(yi)

## dag pipeline edge
### 图中串图
### 大图
+ Parameter Validation Phase
+ Mark Predecessors And Successors Phase
  + 为其每个子节点（节点或者图）修改并行属性
    + 当其子节点已有并行属性，不修改，否则修改
+ construct edge

## 发版本前的工作
+ 代码review
  + base - 已完成
  + device - 已完成
  + thread_pool - 已完成
  + inference - 已完成
  + dag
    + 图里面嵌入图，在释放资源会出错吗？
      + 应该不会，Node和Edge外面包了一层，带有资源管理的字段
    + 边里面要不要加一个字段
      + 输入边 - 没有生产者
      + 中间节点的输入边 - 有消费者且不是整张图的输出
      + 输出边 - 没有消费者，且是整张图的输出
      + 输出边 也是 中间节点的输入边 - 由消费者 且 是整张图的的输出
  + model



## 算子讨论（与守夜大佬讨论）

1. Op的分类实现
例如 二元算子、nn类算子、数学类算子、激活函数类算子，可以共享一些诸如类型检查之类的函数

+ 1. unay
+ 2. binary
+ elementwise
+ nn - compute
+ shape - concat

1. 注册机制
根据设备进行op注册
+ op_type + device （只保留这个）

## 在op内部区别（不是编译、也不是多个子类 - 最传统的函数的方式，if else）
+ data_type + data_format（一两钟实现）

+ opencl - image2d_t buffer

1. 属性层
为每个Op制定attrs（param），用于统一的参数传递，也便于类型检查（tvm的方式）

1. 类型检查
类型检查包含数据类型的合法性（在某个设备上是否支持） 
推导输出的shape大小和维度  relay 

init时调用

5. 计算层（我也觉得）
假设一切输入、属性、内存都是合法的，只专注计算

6. 调用方式（）
如何提供一个明确的接口，可以方便的看出需要进行的参数传递和内容、顺序

relay.nn.conv2d([inputs],kernel,stride)

## 映射关系
conv2d_attrs={

}
relay.call("conv2d",conv2d_attrs)

7. 内存管理
op与edge如何协同  不存在
net op tensor


修改：（这个看你设计）
每一个算子都需要一个头文件吗？？
将参数统一为vector<tensor> inputs, vector<tensor> outputs
weights是否作为inputs中的一员传递 (统一接口层)

attrs作为具体计算类的一个属性赋值


## 后续规划
+ 推理子模块
+ 框架完善
+ python导出

## 模型文件
+ text -> json (rapidjson)
+ bin -> safetensors ()

## llama
### 构图
+ 模型文件 - llama.safetensors
+ 模型结构
  + llama.json
  + expr机制
+ 注：onnx会把模型结构切的很散
### 增加llama ir（rmsnorm\attention\matmul\...）
+ D:\github\nndeploy\framework\include\nndeploy\ir\op_param.h
 + OpType
 + NNDEPLOY_CC_API std::string opTypeToString(OpType op_type);
 + NNDEPLOY_CC_API OpType stringToOpType(const std::string &op_type_name);
 + OpParam - onnx/pytorch/mindspore
 + REGISTER_OP_PARAM_IMPLEMENTION
### op - data_type[fp32(优先这个为主),fp16,int8]/data_format[kDataFormatAuto]
+ cpu op
+ cuda op
+ ascend op
### 内存管理
+ kv block

## llama讨论
+ onnx llama 算子很碎
  + selfattention 十几个算子吗
  + 图优化
+ 手动构图 - 通过expr构造模型结构
  + fp32
  + fp16
  + int8
  + 权重就是safetensors
