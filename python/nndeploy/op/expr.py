from nndeploy.ir import ModelDesc
import nndeploy
import numpy as np
import nndeploy._nndeploy_internal as _C
from nndeploy.base import device_name_to_code

from nndeploy.device import createTensorFromNumpy

"""
该类用于在Python端手动构建计算图
核心是调用了CPP端的makeConv、makeRelu等一系列手动建图接口

"""

class Module:
    def __init__(self):

        self.model_desc = None  # 该Module所属的ModelDesc
        self.weight_map = None  # 记录模型权重名字与Tensor的对应

    def makeExpr(self):
        raise NotImplementedError()


class Conv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=[1, 1],
        groups=1,
    ):
        super().__init__()
        self.param = _C.ir.ConvParam()  # TODO: 构造ConvParam 暂时只设置weight Shape相关
        self.param.kernel_shape_ = [
            out_channels,
            in_channels,
            kernel_size[0],
            kernel_size[1],
        ]
        self.param.group_ = groups

        self.param.dilations_ = dilation

        self.weight_map = [{"weight": None}, {"bias": None}]

    def __call__(self, data):
        return self.makeExpr(data)

    def makeExpr(self, data):

        names = [key for dic in self.weight_map for key in dic.keys()]
        return _C.op.makeConv(
            self.model_desc,
            data,
            self.param,
            names[0],
            names[1],
            names[0],
            names[1],
        )

    def generateWeight(self):
        weight = createTensorFromNumpy(
            np.random.random(self.param.kernel_shape_).astype(np.float32)
        )
        bias = createTensorFromNumpy(
            np.random.random(self.param.kernel_shape_[0]).astype(dtype=np.float32)
        )
        self.weight_map[0]["weight"] = weight
        self.weight_map[1]["bias"] = bias
        return self.weight_map


class Relu(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return self.makeExpr(data)

    def makeExpr(self, data):
        return _C.op.makeRelu(self.model_desc, data, "1", "2")
