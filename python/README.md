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
