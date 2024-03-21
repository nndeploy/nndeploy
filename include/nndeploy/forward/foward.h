
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API NNForwad : public op::NNOp {
 public:
  NNForwad(const std::string &name, op::NNOpType op_type,
           base::DeviceType device_type);

  virtual ~NNForwad();

  virtual base::Status construct(std::vector<std::string> &model_value);

  virtual base::Status init(std::vector<device::Tensor *> inputs,
                            std::vector<device::Tensor *> outputs);

  virtual base::Status deinit();

  virtual base::Status run();

  virtual base::Status backward() = 0;

  virtual base::Status update() = 0;

  virtual base::Status setParam(base::Param *param) = 0;

  virtual base::Param *getParam() = 0;

  virtual base::Status setParallelType(
      const base::ParallelType &paralle_type) = 0;

  virtual base::ParallelType getParallelType() = 0;

  virtual void setInnerFlag(bool flag) = 0;

  virtual void setInitializedFlag(bool flag) = 0;

  virtual bool getInitialized() = 0;

  virtual void setTimeProfileFlag(bool flag) = 0;

  virtual bool getTimeProfileFlag() = 0;

  virtual void setDebugFlag(bool flag) = 0;

  virtual bool getDebugFlag() = 0;

  virtual void setRunningFlag(bool flag) = 0;

  virtual bool isRunning() = 0;

 protected:
  std::vector<device::Tensor *> inputs_;
  std::vector<device::Tensor *> outputs_;
};

}  // namespace forward
}  // namespace nndeploy

#endif