# python

## build & install

1. 开启编译选项

+ 确保使用`git submodule init && git submodule update`下载了所有子模块

+ 在`build/config.cmake`将`ENABLE_NNDEPLOY_PYTHON`设置为`ON`

+ 参照`nndeploy`的构建文档构建完成，执行`make install`

2. 安装Python包

```bash
cd nndeploy/python
pip install -e .  # 开发者模式安装
# 修改为自己的路径
## 连接libnndeploy_xxx.so
export LD_LIBRARY_PATH=/home/always/github/public/nndeploy/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ascenduserdg01/github/nndeploy/build:$LD_LIBRARY_PATH
## 连接onnxruntime
export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH
```

## 接口

Python接口设计与CPP端保持一致，CPP端命名空间体现为Python的Module，例如
+ nndeploy::device::Tensor -> nndeploy.device.Tensor
+ nndeploy::base::DeviceTypeCode  -> nndeploy.base.DeviceTypeCode  

## 功能

### Numpy

+ 支持nndeploy.device.Tensor与numpy array互相转换
    + 维度支持：1、2、3、4
    + 数据类型支持：float32、float16
    + 设备支持：CPU(零拷贝)
    + 限制： 内存为连续

+ 接口设计
    + numpy array 到 Tensor: tensor = nndeploy.device.Tensor(np\_array, nndeploy.base.DeviceTypeCode)
    + Tensor到numpy array： np\_array = numpy.array(tensor)


## TODO

+ 算子接口的导出与测试
+ CUDA设备的支持

## @守夜大佬，在src增加了framework以及plugin的底下的目录

### 讨论
+ 纯c接口
    + 纯c接口写在include/source中，其中需要对外导出的纯c接口，在include中导出c_xxx.h
    + 纯c接口的写法为 nndeployBaseGetDevice()

+ python接口
    + 只包含pybind相关


# pybind11的一些注意点

## 智能指针


https://github.com/pybind/pybind11/issues/956

pybind11默认的pybind class持有类型是`std::unique_ptr`，不能被转换为`std::shared_ptr`。如果报错:

```
RuntimeError: Unable to load a custom holder type from a default-holder instance
```

则可能是将某一返回`std::shared_ptr`的函数，该类没有更改为`std::shared_ptr`，例如`makeExprxx`系列函数，返回`std::shared_ptr<Expr>`，需要将`Expr`的holder显式注册。例如：

```
py::class_<op::Expr,std::shared_ptr<op::Expr>>(m, "Expr")
```

如果修改了还报错，将所有返回std::shared_ptr的函数的返回值管理策略改为` py::return_value_policy::reference`

如何查看是哪个class的问题？
使用python和cpp联合调试，看代码崩溃的栈调用，会显示死在哪个class的析构上

## python/src - cpp通过pybind导出

- 从pybind11导出那层开始 - 遵循python的风格（拿捏不准的可以问gpt，google的python的风格。pytorch也是google风格）
    - 当保持跟cpp一致时，有些类的继承会十分怪异 
    - 例如：
    ```python
    class VulkanArchitecture(_C.device.Architecture):
    def __init__(self):
        super().__init__(common.DeviceTypeCode.vulkan.value)        
        
    def checkDevice(self, device_id=0, library_path=""):
        print("checkDevice")
        return common.Status(common.StatusCode.Ok)

    def enableDevice(self, device_id=0, library_path=""):
        print("enableDevice")
        return common.Status(common.StatusCode.Ok)
    
    def getDevice(self, device_id):
        return VulkanDevice()
    
    def getDeviceInfo(self, library_path=""):
        print("getDeviceInfo")
        return []
    ```

- pybind导出的代码，默认不给外部用户使用吗
    - 故所有需要对外导出的代码，都需要增加一批python的接口

- pybind下的文件保持跟cpp一致
    - 主目录和子目录下的文件（通过cmake来确定那些文件参与编译导出）

- pybind与cpp保持一致的namespace

## python/nndeploy - 纯python代码

- 可以依赖pybind11导出的代码，但是只能nndeploy/xx 依赖对应src/xx 目录下的代码，例如nndeploy/device 依赖对应src/device 目录下的代码，不能依赖 src/base，假如要依赖的化通过依赖nndeploy/base，跟cpp一个逻辑，单向依赖，其中依赖关系如下：
    - framework->plugin
    - framework == base->thread_pool->device->ir->op->net->inference->dag
    - plugin（基本都是独立模块）

- 纯python代码的命名空间与cpp保持一致，便于开发\调试\理解
    - 例如：nndeploy::base::DeviceTypeCode  -> nndeploy.base.DeviceTypeCode
    - 例如：nndeploy::device::Tensor  -> nndeploy.device.Tensor

- cpp的枚举导出，python侧继承，传参更方便

- python侧对于大类，继承+门面模式
