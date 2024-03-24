
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/op/op.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API NNForwad : public op::NNOp {
 public:
  NNForwad(const std::string &name, NNOpType op_type,
           base::DeviceType device_type, interpreter::Interpreter *interpreter,
           std::vector<std::string> &weight_key,
           std::vector<device::Tensor *> inputs,
           std::vector<device::Tensor *> outputs);

  virtual ~NNForwad();

  device::Tensor *createTensor(const std::string &name);

  TensorWrapper *addTensor(device::Tensor *tensor);

  op::NNOp *createOp(const std::string &name, NNOpType op_type,
                     base::DeviceType device_type,
                     interpreter::Interpreter *interpreter,
                     std::vector<std::string> &weight_key,
                     std::vector<device::Tensor *> inputs,
                     std::vector<device::Tensor *> outputs);

  NNOpWrapper *addOp(op::NNOp *op);

  virtual base::Status init();

  virtual base::Status deinit();

  virtual base::Status run();

 protected:
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<NNOpWrapper *> nnop_repository_;
};

NNForwad *createNNForward(const std::string &name, NNOpType op_type,
                          base::DeviceType device_type,
                          interpreter::Interpreter *interpreter,
                          std::vector<std::string> &weight_key,
                          std::vector<device::Tensor *> inputs,
                          std::vector<device::Tensor *> outputs) {
  NNForwad *llama = NNForwad(name, op_type, device_type, interpreter,
                             weight_key, inputs, outputs);

  device::Tensor *attention_0_output =
      llama->createTensor("attention_0_output");
  device::Tensor *attention_0_op =
      llama->createOp("attention_0_op", kNNOpTypeAttention, device_type,
                      interpreter, weight_key, inputs, attention_0_output);

  llama->init();

  return llama;
}

base::Status deleteNNForward(NNForwad *forward) {
  base::Status status = forward->deinit();
  delete forward;

  return status;
}

}  // namespace forward
}  // namespace nndeploy

#endif