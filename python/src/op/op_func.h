#ifndef _NNDEPLOY_PYTHON_SRC_OP_OP_FUNC_H_
#define _NNDEPLOY_PYTHON_SRC_OP_OP_FUNC_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nndeploy/ir/op_param.h"
#include "nndeploy/op/op_add.h"
#include "nndeploy/op/op_batchnorm.h"
#include "nndeploy/op/op_conv.h"
#include "nndeploy/op/op_concat.h"
#include "nndeploy/op/op_flatten.h"
#include "nndeploy/op/op_gemm.h"
#include "nndeploy/op/op_gather.h"
#include "nndeploy/op/op_global_averagepool.h"
#include "nndeploy/op/op_maxpool.h"
#include "nndeploy/op/op_mul.h"
#include "nndeploy/op/op_mat_mul.h"
#include "nndeploy/op/op_relu.h"
#include "nndeploy/op/op_gelu.h"
#include "nndeploy/op/op_rmsnorm.h"
#include "nndeploy/op/op_reshape.h"
#include "nndeploy/op/op_softmax.h"
#include "nndeploy/op/op_slice.h"
#include "nndeploy/op/op_sigmoid.h"
#include "nndeploy/op/op_quantize_linear.h"
#include "nndeploy/op/op_dequantize_linear.h"
#include "nndeploy/op/op_qlinear_conv.h"
#include "nndeploy/op/op_where.h"
#include "nndeploy/op/op_transpose.h"

/**
 * @brief Op的func层，在该层进行Op的输入检查、输出Tensor构造、调用Op计算;
 * 注意：输出Tensor的内存可以不分配，由Op执行函数分配
 */

namespace nndeploy {

device::Tensor* rmsNormFunc(device::Tensor* input, device::Tensor* weight,
                            std::shared_ptr<ir::RMSNormParam> param);

device::Tensor* convFunc(device::Tensor* input, device::Tensor* weight,
                         device::Tensor* bias,
                         std::shared_ptr<ir::ConvParam> param);

device::Tensor* concatFunc(std::vector<device::Tensor *> inputs,
                         std::shared_ptr<ir::ConcatParam> param);

device::Tensor* batchNormFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* shift,
    device::Tensor* mean, device::Tensor* var,
    std::shared_ptr<ir::BatchNormalizationParam> param);

device::Tensor* reluFunc(device::Tensor* input);

device::Tensor* reshapeFunc(device::Tensor* input, device::Tensor* shape,
                            std::shared_ptr<ir::ReshapeParam> param);

device::Tensor* addFunc(device::Tensor* input1, device::Tensor* input2);

device::Tensor* flattenFunc(device::Tensor* input,
                            std::shared_ptr<ir::FlattenParam> param);

device::Tensor* gatherFunc(device::Tensor* input, device::Tensor* index, 
                           std::shared_ptr<ir::GatherParam> param);

device::Tensor* gemmFunc(device::Tensor* inputs_a, device::Tensor* inputs_b,
                         device::Tensor* inputs_c,
                         std::shared_ptr<ir::GemmParam> param);

device::Tensor* geluFunc(device::Tensor* input);
device::Tensor* globalAveragepoolFunc(device::Tensor* input);

device::Tensor* maxPoolFunc(device::Tensor* input,
                            std::shared_ptr<ir::MaxPoolParam> param);

device::Tensor* mulFunc(device::Tensor* input1, device::Tensor* input2);

device::Tensor* matMulFunc(device::Tensor* input1, device::Tensor* input2, 
                           std::shared_ptr<ir::MatMulParam> param, device::Tensor* bias);
device::Tensor* softmaxFunc(device::Tensor* input1,
                            std::shared_ptr<ir::SoftmaxParam> param);
device::Tensor* sigmoidFunc(device::Tensor* input1);
device::Tensor* quantizeLinearFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    std::shared_ptr<ir::QuantizeLinearParam> param);

device::Tensor* dequantizeLinearFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    std::shared_ptr<ir::DequantizeLinearParam> param);

device::Tensor* qlinearConvFunc(
    device::Tensor* x, device::Tensor* x_scale, device::Tensor* x_zero_point,
    device::Tensor* w, device::Tensor* w_scale, device::Tensor* w_zero_point,
    device::Tensor* y_scale, device::Tensor* y_zero_point, device::Tensor* B,
    std::shared_ptr<ir::QLinearConvParam> param);

device::Tensor* whereFunc(device::Tensor* input1, device::Tensor* input2, device::Tensor* condition);
device::Tensor* transposeFunc(device::Tensor* input, std::shared_ptr<ir::TransposeParam> param);
device::Tensor* sliceFunc(
    device::Tensor* input, device::Tensor* starts, device::Tensor* ends,
    device::Tensor* axes, device::Tensor* steps);
}  // namespace nndeploy

#endif
