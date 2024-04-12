
#include "nndeploy/forward/forward.h"

#include "nndeploy/forward/runtime.h"

namespace nndeploy {
namespace forward {

Forwad::Forwad(base::DeviceType device_type, const std::string &name,
               op::OpType op_type)
    : op::Op(device_type, name, op_type) {}
Forwad::~Forwad() {
  for (auto op_wrapper : op_repository_) {
    if (!op_wrapper->is_external_) {
      delete op_wrapper->op_;
    }
    delete op_wrapper;
  }
  for (auto tensor_wrapper : tensor_repository_) {
    if (!tensor_wrapper->is_external_) {
      delete tensor_wrapper->tensor_;
    }
    delete tensor_wrapper;
  }
  op_repository_.clear();
  tensor_repository_.clear();
}

device::Tensor *Forwad::createTensor(const std::string &name) {
  device::Tensor *tensor = new device::Tensor(name);
  TensorWrapper *tensor_wrapper = new TensorWrapper();
  tensor_wrapper->is_external_ = false;
  tensor_wrapper->tensor_ = tensor;
  tensor_wrapper->name_ = name;
  tensor_repository_.emplace_back(tensor_wrapper);
  return tensor;
}

TensorWrapper *Forwad::addTensor(device::Tensor *tensor, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(tensor, "tensor is null!");
  TensorWrapper *tensor_wrapper = new TensorWrapper();
  tensor_wrapper->is_external_ = is_external;
  tensor_wrapper->tensor_ = tensor;
  tensor_wrapper->name_ = tensor->getName();
  tensor_repository_.emplace_back(tensor_wrapper);
  return tensor_wrapper;
}

device::Tensor *Forwad::getTensor(const std::string &name) {
  for (TensorWrapper *tensor_wrapper : tensor_repository_) {
    if (tensor_wrapper->name_ == name) {
      return tensor_wrapper->tensor_;
    }
  }
  return nullptr;
}

op::Op *Forwad::createOp(base::DeviceType device_type, const std::string &name,
                         op::OpType op_type,
                         std::initializer_list<std::string> inputs,
                         std::initializer_list<std::string> outputs,
                         std::initializer_list<std::string> weights) {
  op::Op *op = createOp(device_type, name, op_type, inputs, outputs, weights);
  OpWrapper *op_wrapper = new OpWrapper();
  op_wrapper->is_external_ = false;
  op_wrapper->op_ = op;
  op_wrapper->name_ = name;
  for (auto input : inputs) {
    TensorWrapper *input_wrapper = findTensorWrapper(tensor_repository_, input);
    if (input_wrapper == nullptr) {
      device::Tensor *tensor = this->createTensor(input);
      input_wrapper = this->addTensor(tensor, false);
    }
    input_wrapper->consumers_.emplace_back(op_wrapper);
  }
  for (auto output : outputs) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output);
    if (output_wrapper == nullptr) {
      device::Tensor *tensor = this->createTensor(output);
      output_wrapper = this->addTensor(tensor, false);
    }
    output_wrapper->producers_.emplace_back(op_wrapper);
  }

  op_repository_.emplace_back(op_wrapper);
  return op;
}
op::Op *Forwad::createOp(base::DeviceType device_type, const std::string &name,
                         op::OpType op_type, std::vector<std::string> &inputs,
                         std::vector<std::string> &outputs,
                         std::vector<std::string> &weights) {
  op::Op *op = createOp(device_type, name, op_type, inputs, outputs, weights);
  OpWrapper *op_wrapper = new OpWrapper();
  op_wrapper->is_external_ = false;
  op_wrapper->op_ = op;
  op_wrapper->name_ = name;
  for (auto input : inputs) {
    TensorWrapper *input_wrapper = findTensorWrapper(tensor_repository_, input);
    if (input_wrapper == nullptr) {
      device::Tensor *tensor = this->createTensor(input);
      input_wrapper = this->addTensor(tensor, false);
    }
    input_wrapper->consumers_.emplace_back(op_wrapper);
  }
  for (auto output : outputs) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output);
    if (output_wrapper == nullptr) {
      device::Tensor *tensor = this->createTensor(output);
      output_wrapper = this->addTensor(tensor, false);
    }
    output_wrapper->producers_.emplace_back(op_wrapper);
  }

  op_repository_.emplace_back(op_wrapper);
  return op;
}

base::Status Forwad::addOp(op::Op *op, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(op, "op is null!");
  OpWrapper *op_wrapper = new OpWrapper();
  op_wrapper->is_external_ = is_external;
  op_wrapper->op_ = op;
  op_wrapper->name_ = op->getName();
  for (auto input : op->getAllInput()) {
    TensorWrapper *input_wrapper = findTensorWrapper(tensor_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addTensor(input, is_external);  // todo
    }
    input_wrapper->consumers_.emplace_back(op_wrapper);
  }
  for (auto output : op->getAllOutput()) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addTensor(output, is_external);
    }
    output_wrapper->producers_.emplace_back(op_wrapper);
  }

  op_repository_.emplace_back(op_wrapper);
  return status;
}

base::Status Forwad::setOpParam(const std::string &op_name,
                                base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  OpWrapper *op_wrapper = findOpWrapper(op_repository_, op_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(op_wrapper, "op_wrapper is null!");
  status = op_wrapper->op_->setParam(param);
  return status;
}

base::Param *Forwad::getOpParam(const std::string &op_name) {
  OpWrapper *op_wrapper = findOpWrapper(op_repository_, op_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(op_wrapper, "op_wrapper is null!");
  return op_wrapper->op_->getParam();
}

base::Status Forwad::init() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  status = this->construct();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "graph construct failed!");

  status = this->runtime();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "graph runtime failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(true);

  return status;
}

base::Status Forwad::deinit() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = runtime_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime deinit failed!");
  return status;
}

base::Status Forwad::run() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = runtime_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime run failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(false);

  return status;
}

// base::Status Forwad::dump(std::ostream &oss) {
//   // base::Status status = dumpForward(tensor_repository_, op_repository_,
//   // inputs_,
//   // //                                   outputs_, name_, oss);
//   // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "dump failed!");
//   // return status;
//   return base::kStatusCodeOk;
// }

base::Status Forwad::construct() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type!\n");
  // NNDEPLOY_LOGI("###########################\n");
  base::ParallelType parallel_type = parallel_type_;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto op_wrapper : op_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(op_wrapper->op_,
                                         "tensor_repository_ op is null!");
  }
  for (auto tensor_wrapper : tensor_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(tensor_wrapper->tensor_,
                                         "tensor_repository_ tensor is null!");
    if (tensor_wrapper->producers_.empty() &&
        tensor_wrapper->consumers_.empty()) {
      NNDEPLOY_LOGI("this tensor[%s] is unuseless!\n",
                    tensor_wrapper->tensor_->getName().c_str());
    }
  }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto op_wrapper : op_repository_) {
    Op *op = op_wrapper->op_;
    op->setParallelType(parallel_type);
    op->setInnerFlag(true);
    std::vector<device::Tensor *> inputs = op->getAllInput();
    for (auto input : inputs) {
      TensorWrapper *input_wrapper =
          findTensorWrapper(tensor_repository_, input);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "input_wrapper is null!");

      for (auto producer : input_wrapper->producers_) {
        insertUnique(op_wrapper->predecessors_, producer);
      }
    }
    std::vector<device::Tensor *> outputs = op->getAllOutput();
    for (auto output : outputs) {
      TensorWrapper *output_wrapper =
          findTensorWrapper(tensor_repository_, output);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "output_wrapper is null!");

      for (auto consumer : output_wrapper->consumers_) {
        insertUnique(op_wrapper->successors_, consumer);
      }
    }
  }

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("construct tensor\n");
  // NNDEPLOY_LOGI("##############\n");
  for (auto tensor_wrapper : tensor_repository_) {
    std::vector<Op *> producers;
    for (auto producer : tensor_wrapper->producers_) {
      producers.emplace_back(producer->op_);
    }
    std::vector<Op *> consumers;
    for (auto consumer : tensor_wrapper->consumers_) {
      consumers.emplace_back(consumer->op_);
    }
    // base::Status status =
    //     tensor_wrapper->tensor_->setParallelType(parallel_type);
    // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                        "setParallelType failed!");
    // // 必须在abstract_tensor管理该字段
    // status = tensor_wrapper->tensor_->increaseProducers(producers);
    // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                        "increaseProducers failed!");
    // status = tensor_wrapper->tensor_->increaseConsumers(consumers);
    // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                        "increaseConsumers failed!");
    // status = tensor_wrapper->tensor_->construct();
    // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                        "construct tensor failed!");
  }

  // if (!is_inner_) {
  //   for (auto iter : outputs_) {
  //     iter->markGraphOutput();
  //   }
  // }

  return status;
}

base::Status Forwad::runtime() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type!\n");
  // NNDEPLOY_LOGI("###########################\n");
  base::ParallelType parallel_type = parallel_type_;

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("create runtime\n");
  // NNDEPLOY_LOGI("##############\n");
  // if (parallel_type == base::kParallelTypeNone) {
  //   runtime_ = std::make_shared<SequentialExecutor>();
  // } else if (parallel_type == base::kParallelTypeSequential) {
  //   runtime_ = std::make_shared<SequentialExecutor>();
  // } else if (parallel_type == base::kParallelTypeTask) {
  //   runtime_ = std::make_shared<ParallelTaskExecutor>();
  // } else if (parallel_type == base::kParallelTypePipeline) {
  //   runtime_ = std::make_shared<ParallelPipelineExecutor>();
  // } else {
  //   NNDEPLOY_LOGE("parallel_type is invalid!\n");
  //   return base::kStatusCodeErrorInvalidValue;
  // }
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(runtime_, "Create runtime failed!");

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("runtime init\n");
  // NNDEPLOY_LOGI("1. Optimizer Forwad V1!\n");
  // NNDEPLOY_LOGI("2. Device Verification Phase!\n");
  // NNDEPLOY_LOGI("3. Optimizer Forwad V2!\n");
  // NNDEPLOY_LOGI("4. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("5. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = runtime_->init(tensor_repository_, op_repository_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime init failed!");

  return status;
}

Forwad *createForward(const op::ModelDesc &model_desc,
                      base::DeviceType device_type,
                      base::PrecisionType precision_type) {
  return nullptr;
}

}  // namespace forward
}  // namespace nndeploy
