# 开发者互相讨论

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

## mdc （这块写的很完善，不仅增加了推理框架mdc，还增加了设备mdc，我不是很懂这块呀，大佬这是我这边的一些疑惑呀，方便约个时间讨论下吗）
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