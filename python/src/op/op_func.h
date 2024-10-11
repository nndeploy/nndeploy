#ifndef _PYTHON_SRC_OP_OP_FUNC_H_
#define _PYTHON_SRC_OP_OP_FUNC_H_

#include "nndeploy/op/op_rmsnorm.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

/**
 * @brief Op的func层，在该层进行Op的输入检查、输出Tensor构造、调用Op计算;
 * 注意：输出Tensor的内存可以不分配，由Op执行函数分配
 */

namespace nndeploy {

device::Tensor* rmsNormFunc(device::Tensor* input1, device::Tensor* input2,
                            device::Tensor* input3);
}

#endif
