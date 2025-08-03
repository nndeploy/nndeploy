from nndeploy.ir import ModelDesc
import nndeploy
import numpy as np
import nndeploy._nndeploy_internal as _C
from nndeploy.base import name_to_device_type_code

from nndeploy.device import Tensor
from nndeploy.net.module import Module
import nndeploy.op.functional as F

Expr = _C.op.Expr


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
        weight_name="conv.weight",
        bias_name="conv.bias",
    ):
        super().__init__()
        self.param = _C.ir.ConvParam()
        self.param.kernel_shape_ = [kernel_size[0], kernel_size[1]]
        self.param.group_ = groups
        self.param.dilations_ = dilation
        self.param.pads_ = [padding] * 4
        self.param.strides_ = [stride] * 2

        self.weight_name = weight_name
        self.bias_name = bias_name
        self.weight_map = {weight_name: None, bias_name: None}

    def forward(self, data):
        return Tensor(
            _C.op.conv(
                data,
                self.weight_map[self.weight_name],
                None if self.bias_name is None else self.weight_map[self.bias_name],
                self.param,
            )
        )

    def makeExpr(self, data):
        return _C.op.makeConv(
            self.model_desc, data, self.param, self.weight_name, self.bias_name
        )


class Relu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return Tensor(_C.op.relu(data))

    def makeExpr(self, data):
        return _C.op.makeRelu(self.model_desc, data)


class Add(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return Tensor(_C.op.add(x, y))

    def makeExpr(self, x, y):
        return _C.op.makeAdd(self.model_desc, x, y)


class BatchNorm(Module):
    def __init__(
        self, scale_name, bias_name, mean_name, var_name, eps=1e-5, momentum=0.9
    ):
        super().__init__()
        self.param = _C.ir.BatchNormalizationParam()
        self.param.epsilon_ = eps
        self.param.momentum_ = momentum
        self.scale_name = scale_name
        self.bias_name = bias_name
        self.mean_name = mean_name
        self.var_name = var_name
        self.weight_map = {
            scale_name: None,
            bias_name: None,
            mean_name: None,
            var_name: None,
        }

    def forward(self, data):
        return Tensor(
            _C.op.batch_norm(
                data,
                self.weight_map[self.scale_name],
                self.weight_map[self.bias_name],
                self.weight_map[self.mean_name],
                self.weight_map[self.var_name],
                self.param,
            )
        )

    def makeExpr(self, data):
        return _C.op.makeBatchNorm(
            self.model_desc,
            data,
            self.param,
            self.scale_name,
            self.bias_name,
            self.mean_name,
            self.var_name,
        )


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.param = _C.ir.SoftmaxParam()
        self.param.axis_ = axis

    def forward(self, data):
        return Tensor(_C.op.softmax(data, self.param))

    def makeExpr(self, data):
        return _C.op.makeSoftmax(self.model_desc, data, self.param)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return Tensor(_C.op.sigmoid(data))

    def makeExpr(self, data):
        return _C.op.makeSigmoid(self.model_desc, data)


class Gemm(Module):
    def __init__(
        self, weight_name, bias_name=None, alpha=1.0, beta=1.0, trans_a=0, trans_b=0
    ):
        super().__init__()
        self.param = _C.ir.GemmParam()
        self.param.alpha_ = alpha
        self.param.beta_ = beta
        self.param.trans_a_ = trans_a
        self.param.trans_b_ = trans_b
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.weight_map = {weight_name: None}
        if bias_name is not None:
            self.weight_map[bias_name] = None

    def forward(self, data):
        bias = None if self.bias_name is None else self.weight_map[self.bias_name]
        return Tensor(
            _C.op.gemm(data, self.weight_map[self.weight_name], bias, self.param)
        )

    def makeExpr(self, data):
        return _C.op.makeGemm(
            self.model_desc, data, self.param, self.weight_name, self.bias_name
        )


class Flatten(Module):
    def __init__(self, axis=1):
        super().__init__()
        self.param = _C.ir.FlattenParam()
        self.param.axis_ = axis

    def forward(self, data):
        return Tensor(_C.op.flatten(data, self.param))

    def makeExpr(self, data):
        return _C.op.makeFlatten(self.model_desc, data, self.param)


class MaxPool(Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        self.param = _C.ir.MaxPoolParam()
        self.param.kernel_shape_ = [kernel_size] * 2
        self.param.strides_ = [stride] * 2
        self.param.pads_ = [padding] * 4
        self.param.dilations_ = [dilation] * 2
        self.param.ceil_mode_ = ceil_mode

    def forward(self, data):
        return Tensor(_C.op.maxpool(data, self.param))

    def makeExpr(self, data):
        return _C.op.makeMaxPool(self.model_desc, data, self.param)


class GlobalAveragePool(Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return Tensor(_C.op.global_averagepool(data))

    def makeExpr(self, data):
        return _C.op.makeGlobalAveragePool(self.model_desc, data)
