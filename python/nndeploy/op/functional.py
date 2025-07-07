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


def concat(input1, input2, axis=0):
    param = _C.ir.ConcatParam()
    param.axis_ = axis
    
    return _C.op.concat([input1, input2], param)


def batch_norm(input, scale, bias, mean, var, epsilon=1e-5):
    param = _C.ir.BatchNormalizationParam()
    param.epsilon_ = epsilon

    return _C.op.batch_norm(input, scale, bias, mean, var, param)


def relu(input):
    return _C.op.relu(input)


def gelu(input):
    return _C.op.gelu(input)


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
    param.kernel_shape_ = [kernel_size] * 2
    param.strides_ = [stride] * 2
    param.pads_ = [padding] * 4
    param.dilations_ = [dilation] * 2
    param.ceil_mode_ = ceil_mode
    return _C.op.maxpool(input, param)


def mat_mul(input1, input2, bias=None, transposeA=False, transposeB=False):
    param = _C.ir.MatMulParam()
    param.transposeA_ = transposeA
    param.transposeB_ = transposeB
    if bias == None:
        return _C.op.mat_mul(input1, input2, param, None)
    return _C.op.mat_mul(input1, input2, param, bias)


def mul(input1, input2):
    return _C.op.mul(input1, input2)


def rms_norm(input, weight, epsilon=1e-5):
    param = _C.ir.RMSNormParam()
    param.epsilon_ = epsilon

    return _C.op.rms_norm(input, weight, param)


def softmax(input, axis=1):
    param = _C.ir.SoftmaxParam()
    param.axis_ = axis
    return _C.op.softmax(input, param)


def sigmoid(input):
    return _C.op.sigmoid(input)


def quantize_linear(input, scale, zero_point, axis=1, saturate=True):
    param = _C.ir.QuantizeLinearParam()
    param.axis_ = axis
    param.saturate_ = saturate
    return _C.op.quantize_linear(input, scale, zero_point, param)


def dequantize_linear(input, scale, zero_point, axis=1):
    param = _C.ir.DequantizeLinearParam()
    param.axis_ = axis
    return _C.op.dequantize_linear(input, scale, zero_point, param)


def qlinear_conv(
    x,
    x_scale,
    x_zero_point,
    w,
    w_scale,
    w_zero_point,
    y_scale,
    y_zero_point,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    param = _C.ir.QLinearConvParam()
    param.kernel_shape_ = [w.shape[2], w.shape[3]]
    param.stride_ = [stride, stride]
    param.padding_ = [padding, padding]
    param.dilation_ = [dilation, dilation]
    param.groups_ = groups
    return _C.op.qlinear_conv(
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        y_scale,
        y_zero_point,
        bias,
        param,
    )


def where(input1, input2, condition):
    return _C.op.where(input1, input2, condition)


def transpose(input, perm_axis):
    param = _C.ir.TransposeParam()
    param.perm_ = perm_axis
    return _C.op.transpose(input, param)