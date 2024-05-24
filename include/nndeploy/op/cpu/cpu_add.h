
#ifndef _NNDEPLOY_OP_CPU_ADD_H_
#define _NNDEPLOY_OP_CPU_ADD_H_

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

class CpuAdd : public BinaryOp {
 public:
  CpuAdd(base::DeviceType device_type, const std::string &name, OpType op_type);

  CpuAdd(base::DeviceType device_type, const std::string &name, OpType op_type,
         std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs,
         std::initializer_list<std::string> weights);

  CpuAdd(base::DeviceType device_type, const std::string &name, OpType op_type,
         std::vector<std::string> &inputs, std::vector<std::string> &outputs,
         std::vector<std::string> &weights);

  virtual ~CpuAdd();

  virtual base::Status run();
};

}  // namespace op
}  // namespace nndeploy

#endif /* D50758E3_CDF5_4211_93B3_9C02589B97EF */
