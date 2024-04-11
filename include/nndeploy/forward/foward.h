
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/forward/util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API Forwad : public op::Op {
 public:
  Forwad(base::DeviceType device_type, const std::string &name,
         op::OpType op_type);

  virtual ~Forwad();

  device::Tensor *createTensor(const std::string &name);
  TensorWrapper *addTensor(device::Tensor *tensor, bool is_external);
  device::Tensor *getTensor(const std::string &name);

  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   op::OpType op_type,
                   std::initializer_list<const std::string &> inputs,
                   std::initializer_list<const std::string &> outputs,
                   std::initializer_list<const std::string &> weights);
  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   op::OpType op_type, std::vector<std::string> &inputs,
                   std::vector<std::string> &outputs,
                   std::vector<std::string> &weights);
  OpWrapper *addOp(op::Op *op, bool is_external);

  base::Status setOpParam(const std::string &op_name, base::Param *param);
  base::Param *getOpParam(const std::string &op_name);

  virtual base::Status setPrecisionType(base::PrecisionType precision_type);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(std::vector<device::Tensor *> inputs);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();

 protected:
  virtual base::Status construct();
  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("runtime init\n");
  // NNDEPLOY_LOGI("1. Optimizer Graph V1!\n");
  // NNDEPLOY_LOGI("2. Device Verification Phase!\n");
  // NNDEPLOY_LOGI("3. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("4. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("5. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  virtual base::Status runtime();

 protected:
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;

  std::shared_ptr<Runtime> runtime_;
};

Forwad *createForward(const op::ModelDesc &model_desc,
                      base::DeviceType device_type,
                      base::PrecisionType precision_type);

}  // namespace forward
}  // namespace nndeploy

#endif