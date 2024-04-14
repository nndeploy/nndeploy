
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/forward/runtime.h"
#include "nndeploy/forward/util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API Forwad : public op::Op {
 public:
  Forwad(base::DeviceType device_type, const std::string &name,
         op::OpType op_type);
  Forwad(base::DeviceType device_type, const std::string &name,
         op::OpType op_type, std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs,
         std::initializer_list<std::string> weights);

  Forwad(base::DeviceType device_type, const std::string &name,
         op::OpType op_type, std::vector<std::string> &inputs,
         std::vector<std::string> &outputs, std::vector<std::string> &weights);

  virtual ~Forwad();

  base::Status setModelDesc(std::shared_ptr<op::ModelDesc> model_desc);

  device::Tensor *createTensor(const std::string &name);
  TensorWrapper *addTensor(device::Tensor *tensor, bool is_external = true);
  device::Tensor *getTensor(const std::string &name);

  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   op::OpType op_type,
                   std::initializer_list<std::string> inputs,
                   std::initializer_list<std::string> outputs,
                   std::initializer_list<std::string> weights);
  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   op::OpType op_type, std::vector<std::string> &inputs,
                   std::vector<std::string> &outputs,
                   std::vector<std::string> &weights);
  base::Status addOp(op::Op *op, bool is_external);

  base::Status setOpParam(const std::string &op_name, base::Param *param);
  base::Param *getOpParam(const std::string &op_name);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();

  base::Status dump(std::ostream &oss);

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
  op::ModelDesc *model_desc_;

  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;

  std::shared_ptr<Runtime> runtime_;
};

Forwad *createForward(op::ModelDesc *model_desc, base::DeviceType device_type,
                      base::PrecisionType precision_type);

}  // namespace forward
}  // namespace nndeploy

#endif