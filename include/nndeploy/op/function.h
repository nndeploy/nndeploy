
#ifndef _NNDEPLOY_OP_FUNCTION_H_
#define _NNDEPLOY_OP_FUNCTION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace device {

device::Tensor &operator+(const device::Tensor &a, const device::Tensor &b);
template <typename T>
device::Tensor &operator+(const T &a, const device::Tensor &b) {
  return device::Tensor();
}
template <typename T>
device::Tensor &operator+(const device::Tensor &a, const T &b) {
  return device::Tensor();
}

device::Tensor &operator-(const device::Tensor &a, const device::Tensor &b);
template <typename T>
device::Tensor &operator-(const T &a, const device::Tensor &b) {
  return device::Tensor();
}
template <typename T>
device::Tensor &operator-(const device::Tensor &a, const T &b) {
  return device::Tensor();
}

device::Tensor &operator*(const device::Tensor &a, const device::Tensor &b);
template <typename T>
device::Tensor &operator*(const T &a, const device::Tensor &b) {
  return device::Tensor();
}
template <typename T>
device::Tensor &operator*(const device::Tensor &a, const T &b) {
  return device::Tensor();
}

device::Tensor &operator/(const device::Tensor &a, const device::Tensor &b);
template <typename T>
device::Tensor &operator/(const T &a, const device::Tensor &b) {
  return device::Tensor();
}
template <typename T>
device::Tensor &operator/(const device::Tensor &a, const T &b) {
  return device::Tensor();
}

}  // namespace device
}  // namespace nndeploy

namespace nndeploy {
namespace op {

base::Status add(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status sub(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status mul(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status div(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output);
base::Status clamp(device::Tensor *input, float min, float max,
                   device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

#endif /* _NNDEPLOY_OP_FUNCTION_H_ */
