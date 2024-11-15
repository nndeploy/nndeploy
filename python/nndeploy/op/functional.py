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
