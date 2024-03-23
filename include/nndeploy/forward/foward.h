
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/op/op.h"

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