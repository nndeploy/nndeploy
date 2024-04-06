
#ifndef _NNDEPLOY_FORWARD_FORWARD_H_
#define _NNDEPLOY_FORWARD_FORWARD_H_

#include "nndeploy/forward/util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API Forwad : public op::Op {
 public:
  Forwad(std::string name, OpType op_type, base::DeviceType device_type);
  Forwad(std::string name, OpType op_type, base::DeviceType device_type,
         std::initializer_list<device::Tensor *> inputs,
         std::initializer_list<device::Tensor *> outputs,
         std::vector<std::string> weight_key = {});

  virtual ~Forwad();

  device::Tensor *createTensor(const std::string &name);
  TensorWrapper *addTensor(device::Tensor *tensor);
  device::Tensor *getTensor(const std::string &name);

  op::Op *createOp(std::string name, OpType op_type,
                   base::DeviceType device_type,
                   std::initializer_list<device::Tensor *> inputs,
                   std::initializer_list<device::Tensor *> outputs,
                   std::vector<std::string> weight_key = {});
  op::Op *createOp(std::string name, OpType op_type,
                   base::DeviceType device_type,
                   std::initializer_list<const std::string &> inputs,
                   std::initializer_list<const std::string &> outputs,
                   std::vector<std::string> weight_key = {});
  op::Op *createOp(std::string name, OpType op_type,
                   base::DeviceType device_type,
                   std::initializer_list<device::Tensor *> inputs,
                   std::initializer_list<const std::string &> outputs,
                   std::vector<std::string> weight_key = {});
  op::Op *createOp(std::string name, OpType op_type,
                   base::DeviceType device_type,
                   std::initializer_list<const std::string &> inputs,
                   std::initializer_list<device::Tensor *> outputs,
                   std::vector<std::string> weight_key = {});
  op::Op *createOp(std::string name, OpType op_type,
                   base::DeviceType device_type, device::Tensor *input,
                   device::Tensor *output,
                   std::vector<std::string> weight_key = {});
  op::Op *createOp(std::string name, OpType op_type,
                   base::DeviceType device_type, const std::string &input,
                   const std::string &output,
                   std::vector<std::string> weight_key = {});
  OpWrapper *addOp(op::Op *op, bool is_external);

  base::Status setOpParam(const std::string &name, base::Param *param);
  base::Param *getOpParam(const std::string &name);

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

Forwad *createForward(const ModelDesc &model_desc);

Forwad *createForward(const std::string &name, OpType op_type,
                      base::DeviceType device_type,
                      interpreter::Interpreter *interpreter,
                      std::vector<std::string> &weight_key,
                      std::vector<device::Tensor *> inputs,
                      std::vector<device::Tensor *> outputs) {
  // Forwad *stable_diffusion = Forwad(name, op_type, device_type, interpreter,
  //                                   weight_key, inputs, outputs);

  // device::Tensor *op_0_output = llama->createTensor("op_0_output");
  // op::Op *op_0 = stable_diffusion->createOp(
  //     "op_0", kOpTypeAttention, device_type, interpreter, {"op_0_weight"},
  //     inputs, attention_0_output);

  // stable_diffusion->init();

  Forwad *stable_diffusion = new Forwad(name, op_type);
  stable_diffusion->createOp(name, op_type, inputs[0]->getName(), "op_0_output",
                             "weight_key_0");
  stable_diffusion->createOp(name, op_type, "op_0_output", "op_1_output",
                             "weight_key_1");
  stable_diffusion->createOp(name, op_type, "op_1_output", "op_2_output",
                             "weight_key_2");

  return stable_diffusion;
}

base::Status deleteForward(Forwad *forward) {
  base::Status status = forward->deinit();
  delete forward;

  return status;
}

int main() {
  std::string model_value = "model_value";
  interpreter::Interpreter *interpreter =
      new interpreter::Interpreter(model_value);
  const std::string &name;
  OpType op_type;
  base::DeviceType device_type;
  interpreter::Interpreter *interpreter;
  std::vector<std::string> &weight_key;
  std::vector<device::Tensor *> inputs;
  std::vector<device::Tensor *> outputs;
  Forwad *sd = createForward(name, op_type, device_type, interpreter,
                             weight_key, inputs, outputs);

  stable_diffusion->setInterpreter(interpreter);
  sd->setDeviceType(device_type);

  sd->init();

  sd->run();

  sd->deinit();

  base::Status status = deleteForward(sd);
}

}  // namespace forward
}  // namespace nndeploy

#endif