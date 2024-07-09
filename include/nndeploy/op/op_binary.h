
#ifndef _NNDEPLOY_OP_OP_BINARY_H_
#define _NNDEPLOY_OP_OP_BINARY_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpBinary : public Op {
 public:
  OpBinary() : Op() {}

  virtual ~OpBinary() {}

  virtual base::Status reshape(base::ShapeMap &shape_map) {
    base::Status status = base::kStatusCodeOk;
    for (auto input : inputs_) {
      std::string name = input->getName();
      if (shape_map.find(name) == shape_map.end()) {
        NNDEPLOY_LOGE("shape_map not found %s", name.c_str());
        return base::kStatusCodeErrorInvalidParam;
      }
      status = input->reshape(shape_map[name]);
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "reshape failed");
    }
    auto iter = shape_map.begin();
    base::IntVector shape0 = iter->second;
    iter++;
    base::IntVector shape1 = iter->second;
    base::IntVector shape_out = shape0;
    if (shape0.size() < shape1.size()) {
      shape_out = shape1;
    }
    status = outputs_[0]->reshape(shape_out);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "reshape failed");
    return status;
  }
};

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

#endif
