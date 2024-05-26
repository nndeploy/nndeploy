
#include "nndeploy/op/function.h"

namespace nndeploy {
namespace device {

device::Tensor &operator+(const device::Tensor &a, const device::Tensor &b) {
  return device::Tensor();
}

device::Tensor &operator-(const device::Tensor &a, const device::Tensor &b) {
  return device::Tensor();
}

device::Tensor &operator*(const device::Tensor &a, const device::Tensor &b) {
  return device::Tensor();
}

device::Tensor &operator/(const device::Tensor &a, const device::Tensor &b) {
  return device::Tensor();
}

}  // namespace op
}  // namespace nndeploy