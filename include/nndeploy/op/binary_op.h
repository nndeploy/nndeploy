
#ifndef _NNDEPLOY_OP_BINARY_ADD_H_
#define _NNDEPLOY_OP_BINARY_ADD_H_

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
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class BinaryOp : public Op {
 public:
  BinaryOp(base::DeviceType device_type, const std::string &name,
           OpType op_type)
      : Op(device_type, name, op_type) {}

  BinaryOp(base::DeviceType device_type, const std::string &name,
           OpType op_type, std::initializer_list<std::string> inputs,
           std::initializer_list<std::string> outputs,
           std::initializer_list<std::string> weights)
      : Op(device_type, name, op_type, inputs, outputs, weights) {}

  BinaryOp(base::DeviceType device_type, const std::string &name,
           OpType op_type, std::vector<std::string> &inputs,
           std::vector<std::string> &outputs, std::vector<std::string> &weights)
      : Op(device_type, name, op_type, inputs, outputs, weights) {}

  virtual ~BinaryOp() {}

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

}  // namespace op
}  // namespace nndeploy

#endif
