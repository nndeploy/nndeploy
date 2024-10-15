# 算子

## 添加新算子

1. 算子定义与声明

nndeploy算子以类的方式注册和使用。算子的实现与后端强绑定，但不同后端的同一类型算子可以共享某些公共函数，例如形状推导。在nndeploy中算子分为两个层级，第一个父类层级在`nndeploy/framework/include/nndeploy/op`声明，在`nndeploy/framework/source/nndeploy/op`中定义。例如`RMSNorm`:

```
class OpRMSNorm : public Op {
 public:
  OpRMSNorm() : Op() { is_inplace_ = true; }
  virtual ~OpRMSNorm() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};


```

父类层级中需要实现两个关键函数：

+ `inferShape`: 对输出Tensor的形状进行推导，这个过程是沿着网络向后传播的；当网络的`input`大小确定后，沿着网络不断推导路径上所有`Tensor`的形状。当形状推导完成后，根据`Tensor`的形状和生命周期进行后续的内存申请。

+ `run`: 父类的`run`函数实现一个不依赖任何特定指令集，在大部分CPU上都能运行的实现函数，作为无优化版本的快速实现。

特定后端的子类`Op`实现在`nndeploy/framework/source/nndeploy/op/xxx`下，`xxx`是后端文件夹。如无特殊需求要重写父类的其他虚函数，则只用实现`run`。在这一层假设所有输入、输出的内存、形状、数值类型都是正确的，只需关注实现。

所有的`Op`必须注册到工厂模式中：

```
REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         ir::kOpTypeSoftmax, OpSoftmax)

```

2. 函数形式算子

以上实现都为类的形式，有时希望以函数形式来对Tensor直接进行计算，因此封装了函数形式的算子直接调用。该函数接口的声明与定义和父类`Op`在相同的文件中。各个算子基本都是固定的形式，创建`Op`对象，设置输入输出，初始化与运行。例如:

```
base::Status rmsNorm(device::Tensor *input1, device::Tensor *input2,
                     device::Tensor *input3, device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input1->getDeviceType(), "", ir::kOpTypeRMSNorm);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setInput(input1, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input2, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input3, 2);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");

  delete op;

  return status;
}


```


3. Python接口导出

在Python端导出函数形式的算子，可以使用`nndeploy.op.xxop(input)`直接调用。导出接口位于`nndeploy/python/src/op/op.cc`中，例如：

```
NNDEPLOY_API_PYBIND11_MODULE("op", m) { m.def("rms_norm", rmsNormFunc); }

```
表示导出到python的`op`模块，`nndeploy.op.rms_norm`将被定向到`rmsNormFunc`函数中执行。`rmsNormFunc`是Python到Cpp接口的一个中间层，其实现在`nndeploy/python/src/op/op_func.cc`中。在Func层进行输入的合法性检查，如果是非inplace算子，则还要申请输出Tensor。但不对输出Tenso进行内存开辟，因为内存开辟依赖于形状推理，这部分在函数算子内部进行。

4. 测试

算子的测试位于`nndeploy/python/nndeploy/tests/op`, 使用`unittest`框架测试。 数据的构造可以使用numpy，然后转为nndeploy的`Tensor`。标准结果的对比可以使用numpy或PyTorch，将nndeploy的Tensor转换为numpy进行对比。


