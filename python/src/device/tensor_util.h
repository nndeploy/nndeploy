#ifndef _NNDEPLOY_PYTHON_SRC_DEVICE_TENSOR_UTIL_H_
#define _NNDEPLOY_PYTHON_SRC_DEVICE_TENSOR_UTIL_H_

#include <pybind11/numpy.h>

#include <string>

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace py = pybind11;

namespace nndeploy {
namespace device {

py::array tensorArray(Tensor &tensor);

// 从numpy初始化Tensor
device::Tensor* bufferInfoToTensorByDeviceTypeCode(const py::buffer& buffer,
                                   const base::DeviceTypeCode& device_type_code);

// 从numpy初始化Tensor
device::Tensor* bufferInfoToTensor(const py::buffer& buffer,
                                   const base::DeviceType& device_type);

// 将Tensor搬移到其他设备上
device::Tensor* moveTensorToDeviceByDeviceTypeCode(device::Tensor* tensor,
                                   const base::DeviceTypeCode& device_type_code);

// 将Tensor搬移到其他设备上
device::Tensor* moveTensorToDevice(device::Tensor* tensor,
                                   const base::DeviceType& device_type);

}  // namespace device
}  // namespace nndeploy

#endif