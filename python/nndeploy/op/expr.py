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
        bias=True,
        weight_name="conv.weight",
        bias_name="conv.bias",
    ):
        super().__init__()
        self.param = _C.ir.ConvParam()  # TODO: 构造ConvParam 暂时只设置weight Shape相关
        self.param.kernel_shape_ = [kernel_size[0], kernel_size[1]]
        # self.param.group_ = groups

        # self.param.dilations_ = dilation

        self.weight_name = weight_name
        self.bias_name = bias_name

        self.weight_map = {weight_name: None, bias_name: None}

    def __call__(
        self,
        data,
    ):
        return self.makeExpr(data)

    def makeExpr(self, data):

        return _C.op.makeConv(
            self.model_desc,
            data,
            self.param,
            self.weight_name,
            self.bias_name,
            "",
            "",
        )


class Relu(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return self.makeExpr(data)

    def makeExpr(self, data):
        return _C.op.makeRelu(self.model_desc, data, "", "")


class BatchNorm(Module):
    def __init__(self, scale_name, bias_name, mean_name, var_name):
        super().__init__()

        self.param = _C.ir.BatchNormalizationParam()
        self.scale_name = scale_name
        self.bias_name = bias_name
        self.mean_name = mean_name
        self.var_name = var_name

        self.weight_map = {
            self.scale_name: None,
            self.bias_name: None,
            self.mean_name: None,
            self.var_name: None,
        }

    def __call__(self, data):
        return self.makeExpr(data)

    def makeExpr(self, data):
        return _C.op.makeBatchNorm(
            self.model_desc,
            data,
            self.param,
            self.scale_name,
            self.bias_name,
            self.mean_name,
            self.var_name,
            "",
            ""
        )
