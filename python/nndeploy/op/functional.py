"""
函数形式Op
"""

import nndeploy._nndeploy_internal as _C


def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert len(weight.shape) == 4  # 当前仅支持OIHW格式的权重
    param = _C.ir.ConvParam()
    param.dilations_ = [dilation, dilation]
    param.group_ = groups
    param.kernel_shape_ = [weight.shape[2], weight.shape[3]]
    param.strides_ = [stride, stride]
    param.pads_ = [padding, padding, padding, padding]

    return _C.op.conv(input, weight, bias, param)


def batch_norm(input, scale, bias, mean, var, epsilon=1e-5):
    param = _C.ir.BatchNormalizationParam()
    param.epsilon_ = epsilon

    return _C.op.batch_norm(input, scale, bias, mean, var, param)


def relu(input):

    return _C.op.relu(input)


def add(input1, input2):
    return _C.op.add(input1, input2)


def flatten(input, axis=1):
    param = _C.ir.FlattenParam()
    param.axis_ = axis
    return _C.op.flatten(input, param)


def gemm(input_a, input_b, input_c=None, alpha=1.0, beta=1.0, trans_a=0, trans_b=0):
    param = _C.ir.GemmParam()
    param.alpha_ = alpha
    param.beta_ = beta
    param.trans_a_ = trans_a
    param.trans_b_ = trans_b
    return _C.op.gemm(input_a, input_b, input_c, param)


def global_averagepool(input):
    return _C.op.global_averagepool(input)


def maxpool(input, kernel_size, stride=1, padding=0, dilation=1, ceil_mode=False):
    param = _C.ir.MaxPoolParam()
    param.kernel_shape_ = [kernel_size] *2
    param.strides_ = [stride] * 2
    param.pads_ = [padding] * 4
    param.dilations_ = [dilation] * 2
    param.ceil_mode_ = ceil_mode
    return _C.op.maxpool(input, param)
