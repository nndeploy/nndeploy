#ifndef _NNDEPLOY_PYTHON_SRC_DEVICE_TENSOR_UTIL_H_
#define _NNDEPLOY_PYTHON_SRC_DEVICE_TENSOR_UTIL_H_

#include <pybind11/pybind11.h>

#include <string>

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace py = pybind11;

namespace nndeploy {
namespace device {

// 从tensor的shape推断stride，numpy要求的内存是紧凑的
// stride按照元素个数计数而非numpy中采用的按照元素Bytes
std::vector<long> calculateStridesBaseShape(const base::IntVector& shape);

// 获取nndepoly::device::Tensor 转 numpy array的必要信息
py::buffer_info tensorToBufferInfo(device::Tensor* tensor);

// 从numpy初始化Tensor
device::Tensor* bufferInfoToTensor(const py::buffer& buffer,
                                   const base::DeviceType& device_type);

// 将Tensor搬移到其他设备上
device::Tensor* moveTensorToDevice(device::Tensor* tensor,
                                   const base::DeviceType& device_type);

}  // namespace device
}  // namespace nndeploy

#endif