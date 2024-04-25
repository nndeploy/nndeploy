# TODO

// Net *createNet(const std::string &name, op::OpType op_type,
//                       base::DeviceType device_type,
//                       interpreter::Interpreter *interpreter,
//                       std::vector<std::string> &weight_key,
//                       std::vector<device::Tensor *> inputs,
//                       std::vector<device::Tensor *> outputs) {
//   // Net *stable_diffusion = Net(name, op_type, device_type,
//   interpreter,
//   //                                   weight_key, inputs, outputs);

//   // device::Tensor *op_0_output = llama->createTensor("op_0_output");
//   // op::Op *op_0 = stable_diffusion->createOp(
//   //     "op_0", kOpTypeAttention, device_type, interpreter, {"op_0_weight"},
//   //     inputs, attention_0_output);

//   // stable_diffusion->init();

//   Net *stable_diffusion = new Net(name, op_type);
//   stable_diffusion->createOp(name, op_type, inputs[0]->getName(),
//   "op_0_output",
//                              "weight_key_0");
//   stable_diffusion->createOp(name, op_type, "op_0_output", "op_1_output",
//                              "weight_key_1");
//   stable_diffusion->createOp(name, op_type, "op_1_output", "op_2_output",
//                              "weight_key_2");

//   return stable_diffusion;
// }

// base::Status deleteNet(Net *net) {
//   base::Status status = net->deinit();
//   delete net;

//   return status;
// }

// int main() {
//   std::string model_value = "model_value";
//   interpreter::Interpreter *interpreter =
//       new interpreter::Interpreter(model_value);
//   const std::string &name;
//   op::OpType op_type;
//   base::DeviceType device_type;
//   interpreter::Interpreter *interpreter;
//   std::vector<std::string> &weight_key;
//   std::vector<device::Tensor *> inputs;
//   std::vector<device::Tensor *> outputs;
//   Net *sd = createNet(name, op_type, device_type, interpreter,
//                              weight_key, inputs, outputs);

//   stable_diffusion->setInterpreter(interpreter);
//   sd->setDeviceType(device_type);

//   sd->init();

//   sd->run();

//   sd->deinit();

//   base::Status status = deleteNet(sd);
// }