#ifndef _NNDEPLOY_PYTHON_SRC_OP_OP_FUNC_H_
#define _NNDEPLOY_PYTHON_SRC_OP_OP_FUNC_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nndeploy/ir/op_param.h"
#include "nndeploy/op/op_add.h"
#include "nndeploy/op/op_batchnorm.h"
#include "nndeploy/op/op_conv.h"
#include "nndeploy/op/op_flatten.h"
#include "nndeploy/op/op_gemm.h"
#include "nndeploy/op/op_global_averagepool.h"
#include "nndeploy/op/op_maxpool.h"
#include "nndeploy/op/op_relu.h"
#include "nndeploy/op/op_rmsnorm.h"

/**
 * @brief Op的func层，在该层进行Op的输入检查、输出Tensor构造、调用Op计算;
 * 注意：输出Tensor的内存可以不分配，由Op执行函数分配
 */

namespace nndeploy {

device::Tensor* rmsNormFunc(device::Tensor* input1, device::Tensor* input2,
                            device::Tensor* input3);

device::Tensor* convFunc(device::Tensor* input, device::Tensor* weight,
                         device::Tensor* bias,
                         std::shared_ptr<ir::ConvParam> param);

device::Tensor* batchNormFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* bias,
    device::Tensor* mean, device::Tensor* var,
    std::shared_ptr<ir::BatchNormalizationParam> param);

device::Tensor* reluFunc(device::Tensor* input);

device::Tensor* addFunc(device::Tensor* input1, device::Tensor* input2);

device::Tensor* flattenFunc(device::Tensor* input,
                            std::shared_ptr<ir::FlattenParam> param);

device::Tensor* gemmFunc(device::Tensor* inputs_a, device::Tensor* inputs_b,
                         device::Tensor* inputs_c,
                         std::shared_ptr<ir::GemmParam> param);

device::Tensor* globalAveragepoolFunc(device::Tensor* input);

device::Tensor* maxPoolFunc(device::Tensor* input,
                            std::shared_ptr<ir::MaxPoolParam> param);

}  // namespace nndeploy

#endif
