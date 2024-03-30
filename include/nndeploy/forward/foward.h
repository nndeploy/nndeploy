
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/op/op.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API NNForwad : public op::Op {
 public:
  NNForwad(const std::string &name, OpType op_type,
           base::DeviceType device_type, interpreter::Interpreter *interpreter,
           std::vector<std::string> &weight_key,
           std::vector<device::Tensor *> inputs,
           std::vector<device::Tensor *> outputs);

  virtual ~NNForwad();

  device::Tensor *createTensor(const std::string &name);

  TensorWrapper *addTensor(device::Tensor *tensor);

  op::Op *createOp(const std::string &name, OpType op_type,
                   base::DeviceType device_type,
                   interpreter::Interpreter *interpreter,
                   std::vector<std::string> &weight_key,
                   std::vector<device::Tensor *> inputs,
                   std::vector<device::Tensor *> outputs);

  NNOpWrapper *addOp(op::Op *op);

  virtual base::Status init();

  virtual base::Status deinit();

  virtual base::Status run();

 protected:
  bool auto_construct_forward_flag_ = false;
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<NNOpWrapper *> nnop_repository_;
};

NNForwad *createNNForward(const std::string &name, OpType op_type,
                          base::DeviceType device_type,
                          interpreter::Interpreter *interpreter,
                          std::vector<std::string> &weight_key,
                          std::vector<device::Tensor *> inputs,
                          std::vector<device::Tensor *> outputs) {
  NNForwad *stable_diffusion = NNForwad(name, op_type, device_type, interpreter,
                                        weight_key, inputs, outputs);

  device::Tensor *op_0_output = llama->createTensor("op_0_output");
  op::Op *op_0 = stable_diffusion->createOp(
      "op_0", kNNOpTypeAttention, device_type, interpreter, {"op_0_weight"},
      inputs, attention_0_output);

  stable_diffusion->init();

  return stable_diffusion;
}

base::Status deleteNNForward(NNForwad *forward) {
  base::Status status = forward->deinit();
  delete forward;

  return status;
}

}  // namespace forward
}  // namespace nndeploy

#endif