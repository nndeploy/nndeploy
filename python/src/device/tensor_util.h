#ifndef _NNDEPLOY_PYTHON_SRC_DEVICE_TENSOR_UTIL_H_
#define _NNDEPLOY_PYTHON_SRC_DEVICE_TENSOR_UTIL_H_

#include <pybind11/pybind11.h>

#include <string>

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

using namespace nndeploy;
namespace py = pybind11;

// 获取tensor的数值类型 根据元素的bit位宽决定
std::string getTensorFormat(device::Tensor* tensor);

// 从tensor的shape推断stride，numpy要求的内存是紧凑的
// stride按照元素个数计数而非numpy中采用的按照元素Bytes
std::vector<long> calculateStridesBaseShape(const base::IntVector& shape);

// 获取nndepoly::device::Tensor 转 numpy array的必要信息
py::buffer_info tensorToBufferInfo(device::Tensor* tensor);

// 从numpy初始化Tensor
device::Tensor* bufferInfoToTensor(py::buffer const b,
                                   base::DeviceTypeCode device_code);

// 将Tensor搬移到其他设备上
device::Tensor* moveTensorToDevice(device::Tensor* tensor,
                                   base::DeviceTypeCode device_code);

#endif