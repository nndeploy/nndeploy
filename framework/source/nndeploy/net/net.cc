
#include "nndeploy/net/net.h"

#include "nndeploy/net/session.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace net {

Net::Net() : op::Op() {}

Net::~Net() {}

base::Status Net::setModelDesc(ir::ModelDesc *model_desc) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(model_desc, "model_desc is null!");
  model_desc_ = model_desc;
  return status;
}

TensorWrapper *Net::createTensor(const std::string &name, bool is_weight) {
  device::Tensor *tensor = new device::Tensor(name);
  TensorWrapper *tensor_wrapper = new TensorWrapper();
  tensor_wrapper->is_external_ = false;
  tensor_wrapper->is_weight_ = is_weight;
  tensor_wrapper->tensor_ = tensor;
  tensor_wrapper->name_ = name;
  tensor_repository_.emplace_back(tensor_wrapper);
  return tensor_wrapper;
}

TensorWrapper *Net::addTensor(device::Tensor *tensor, bool is_external,
                              bool is_weight) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(tensor, "tensor is null!");
  TensorWrapper *tensor_wrapper = new TensorWrapper();
  tensor_wrapper->is_external_ = is_external;
  tensor_wrapper->is_weight_ = is_weight;
  tensor_wrapper->tensor_ = tensor;
  tensor_wrapper->name_ = tensor->getName();
  tensor_repository_.emplace_back(tensor_wrapper);
  return tensor_wrapper;
}

device::Tensor *Net::getTensor(const std::string &name) {
  for (TensorWrapper *tensor_wrapper : tensor_repository_) {
    if (tensor_wrapper->name_ == name) {
      return tensor_wrapper->tensor_;
    }
  }
  return nullptr;
}

bool Net::isWeight(const std::string &name) {
  if (model_desc_->weights_.find(name) != model_desc_->weights_.end()) {
    return true;
  }
  return false;
}
device::Tensor *Net::getWeight(const std::string &weight) {
  device::Tensor *weight_tensor = nullptr;
  if (model_desc_->weights_.find(weight) != model_desc_->weights_.end()) {
    weight_tensor = model_desc_->weights_[weight];
    model_desc_->weights_[weight] = nullptr;
  } else {
    NNDEPLOY_LOGE("weight[%s] is not found!\n", weight.c_str());
  }
  return weight_tensor;
}

op::Op *Net::createOp(base::DeviceType device_type, const std::string &name,
                      ir::OpType op_type,
                      std::initializer_list<std::string> inputs,
                      std::initializer_list<std::string> outputs) {
  // TODO: 这里命名与namespace op下的createOp冲突
  // 必须使用op::  否则递归了
  op::Op *op = op::createOp(device_type, name, op_type, inputs, outputs);
  if (op == nullptr) {
    NNDEPLOY_LOGE("Failed to create Op: %s\n", name.c_str());
    return nullptr;
  }
  OpWrapper *op_wrapper = new OpWrapper();
  op_wrapper->is_external_ = false;
  op_wrapper->op_ = op;
  op_wrapper->name_ = name;
  for (auto input : inputs) {
    TensorWrapper *input_wrapper = findTensorWrapper(tensor_repository_, input);
    if (input_wrapper == nullptr) {
      if (isWeight(input)) {
        device::Tensor *weight = getWeight(input);
        // input_wrapper = new TensorWrapper();
        // input_wrapper->is_external_ = false;
        // input_wrapper->is_weight_ = true;
        // input_wrapper->tensor_ = weight;
        // input_wrapper->name_ = input;
        input_wrapper = this->addTensor(weight, false, true);
      } else {
        input_wrapper = this->createTensor(input);
        if (input_wrapper == nullptr) {
          NNDEPLOY_LOGE("create tensor failed!\n");
          return nullptr;
        }
      }
    }
    op->setInput(input_wrapper->tensor_);
    // input_wrapper->consumers_.emplace_back(op_wrapper);
    insertUnique(input_wrapper->consumers_, op_wrapper);
  }
  for (auto output : outputs) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->createTensor(output);
      if (output_wrapper == nullptr) {
        NNDEPLOY_LOGE("create tensor failed!\n");
        return nullptr;
      }
    }
    op->setOutput(output_wrapper->tensor_);
    // output_wrapper->producers_.emplace_back(op_wrapper);
    insertUnique(output_wrapper->producers_, op_wrapper);
  }

  op_repository_.emplace_back(op_wrapper);
  return op;
}
op::Op *Net::createOp(base::DeviceType device_type, const std::string &name,
                      ir::OpType op_type, std::vector<std::string> &inputs,
                      std::vector<std::string> &outputs) {
  op::Op *op = op::createOp(device_type, name, op_type, inputs, outputs);
  if (op == nullptr) {
    NNDEPLOY_LOGE("Failed to create Op: %s\n", name.c_str());
    return nullptr;
  }
  OpWrapper *op_wrapper = new OpWrapper();
  op_wrapper->is_external_ = false;
  op_wrapper->op_ = op;
  op_wrapper->name_ = name;
  for (auto input : inputs) {
    TensorWrapper *input_wrapper = findTensorWrapper(tensor_repository_, input);
    if (input_wrapper == nullptr) {
      if (isWeight(input)) {
        device::Tensor *weight = getWeight(input);
        // input_wrapper = new TensorWrapper();
        // input_wrapper->is_external_ = false;
        // input_wrapper->is_weight_ = true;
        // input_wrapper->tensor_ = weight;
        // input_wrapper->name_ = input;
        input_wrapper = this->addTensor(weight, false, true);
      } else {
        input_wrapper = this->createTensor(input);
        if (input_wrapper == nullptr) {
          NNDEPLOY_LOGE("create tensor failed!\n");
          return nullptr;
        }
      }
    }
    op->setInput(input_wrapper->tensor_);
    // input_wrapper->consumers_.emplace_back(op_wrapper);
    insertUnique(input_wrapper->consumers_, op_wrapper);
  }
  for (auto output : outputs) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->createTensor(output);
      if (output_wrapper == nullptr) {
        NNDEPLOY_LOGE("create tensor failed!\n");
        return nullptr;
      }
    }
    op->setOutput(output_wrapper->tensor_);
    // output_wrapper->producers_.emplace_back(op_wrapper);
    insertUnique(output_wrapper->producers_, op_wrapper);
  }

  op_repository_.emplace_back(op_wrapper);
  return op;
}

base::Status Net::addNet(Net *net, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(net, "op is null!");
  OpWrapper *op_wrapper = new OpWrapper();
  op_wrapper->is_external_ = is_external;
  op_wrapper->op_ = net;
  op_wrapper->name_ = net->getName();
  for (auto input : net->getAllInput()) {
    TensorWrapper *input_wrapper = findTensorWrapper(tensor_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addTensor(input, is_external);  // todo
    }
    // input_wrapper->consumers_.emplace_back(op_wrapper);
    insertUnique(input_wrapper->consumers_, op_wrapper);
  }
  for (auto output : net->getAllOutput()) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addTensor(output, is_external);
    }
    // output_wrapper->producers_.emplace_back(op_wrapper);
    insertUnique(output_wrapper->producers_, op_wrapper);
  }

  op_repository_.emplace_back(op_wrapper);
  return status;
}

base::Status Net::setOpParam(const std::string &op_name,
                             std::shared_ptr<base::Param> param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  OpWrapper *op_wrapper = findOpWrapper(op_repository_, op_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(op_wrapper, "op_wrapper is null!");
  status = op_wrapper->op_->setParam(param);
  return status;
}

std::shared_ptr<base::Param> Net::getOpParam(const std::string &op_name) {
  OpWrapper *op_wrapper = findOpWrapper(op_repository_, op_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(op_wrapper, "op_wrapper is null!");
  return op_wrapper->op_->getParam();
}

base::Status Net::init() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  status = this->construct();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "graph construct failed!");

  // 即使是设备相关的图优化，也可以放在优化器中做
  // 经过这一次图优化之后
  status = optimizer();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "graph construct failed!");

  status = inferDataType();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "inferDataType failed!");

  status = inferShape();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "inferShape failed!");

  status = this->session();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "graph session failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(true);

  return status;
}

base::Status Net::deinit() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = session_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "session deinit failed!");
  delete session_;
  session_ = nullptr;

  for (auto op_wrapper : op_repository_) {
    if (!op_wrapper->is_external_) {
      delete op_wrapper->op_;
    }
    delete op_wrapper;
  }
  op_repository_.clear();

  for (auto tensor_wrapper : tensor_repository_) {
    if (!tensor_wrapper->is_external_) {
      delete tensor_wrapper->tensor_;
    }
    delete tensor_wrapper;
  }
  tensor_repository_.clear();

  return status;
}

base::Status Net::inferDataType() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->inferDataType();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "inferDataType failed!");
  }
  return status;
};
base::Status Net::inferShape() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    NNDEPLOY_LOGI("Op Name: %s\n", iter->op_->getName().c_str());
    status = iter->op_->inferShape();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "inferDataType failed!");
  }
  return status;
};
base::Status Net::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = session_->reshape(shape_map);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "session preRun failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(false);

  return status;
};

base::Status Net::preRun() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = session_->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "session preRun failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(false);

  return status;
};

base::Status Net::run() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = session_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "session run failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(false);

  return status;
}

base::Status Net::postRun() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = session_->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "session run failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(false);

  return status;
};

base::Status Net::dump(std::ostream &oss) {
  base::Status status = dumpNet(tensor_repository_, op_repository_, inputs_,
                                outputs_, op_desc_.name_, oss);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "dump failed!");
  return base::kStatusCodeOk;
}

/**
 * @brief 遍历ModelDesc中的构图信息，生成TensorWrapper和OpWrapper
 * @return base::Status
 */
base::Status Net::construct() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type!\n");
  // NNDEPLOY_LOGI("###########################\n");
  base::ParallelType parallel_type = parallel_type_;

  // 算子的构建
  for (auto op_desc_ : model_desc_->op_descs_) {
    // 获得每一个Op的输入、输出名字
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (auto input_name : op_desc_->inputs_) {
      insertUnique(input_names, input_name);
      // NNDEPLOY_LOGE("op_desc->inputs_ = %s\n", input_name.c_str());
    }
    for (auto output_name : op_desc_->outputs_) {
      insertUnique(output_names, output_name);
      // NNDEPLOY_LOGE("op_desc->outputs_ = %s\n", output_name.c_str());
    }

    // createOp内部包含了CreateTensor的步骤，且是Unique的
    auto op = this->createOp(device_type_, op_desc_->name_, op_desc_->op_type_,
                             input_names, output_names);
    if (op == nullptr) {
      NNDEPLOY_LOGE("Failed to create Op: %s\n", op_desc_->name_.c_str());
      return base::kStatusCodeErrorInvalidValue;
    }
    // 不对param进行判空检查  有些op没有param，例如relu
    op->setParam(op_desc_->op_param_);
  }

  // 输入的构建
  for (auto input : model_desc_->inputs_) {
    TensorWrapper *input_wrapper =
        findTensorWrapper(tensor_repository_, input->name_);
    if (input_wrapper == nullptr) {
      input_wrapper = this->createTensor(input->name_);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "create tensor failed!");
    }
    inputs_.emplace_back(input_wrapper->tensor_);

    input_wrapper->tensor_->setDataType(input->data_type_);

    base::IntVector shape = input->shape_;
    if (shape.empty()) {
      if (opt_shape_.find(input->name_) != opt_shape_.end()) {
        shape = opt_shape_[input->name_];
      }
    }
    if (is_dynamic_shape_) {
      if (max_shape_.find(input->name_) != max_shape_.end()) {
        shape = max_shape_[input->name_];
      }
    }
    input_wrapper->tensor_->reshape(shape);
  }

  // 输出的构建
  for (auto output : model_desc_->outputs_) {
    TensorWrapper *output_wrapper =
        findTensorWrapper(tensor_repository_, output->name_);
    if (output_wrapper == nullptr) {
      output_wrapper = this->createTensor(output->name_);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "create tensor failed!");
    }
    outputs_.emplace_back(output_wrapper->tensor_);
  }

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto op_wrapper : op_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(op_wrapper->op_,
                                         "tensor_repository_ op is null!");
  }
  for (auto tensor_wrapper : tensor_repository_) {
    NNDEPLOY_LOGI("tensor name = %s\n", tensor_wrapper->name_.c_str());
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
    op->setPrecisionType(precision_type_);
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

  // 拓扑排序
  std::vector<OpWrapper *> topo_op_repository;
  status = topoSortDFS(op_repository_, topo_op_repository);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "topoSortDFS failed!");
  op_repository_.clear();
  op_repository_ = topo_op_repository;

  return status;
}

base::Status Net::optimizer() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status Net::session() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("create session\n");
  // NNDEPLOY_LOGI("##############\n");
  session_ = createSession(device_type_, parallel_type_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(session_, "Create session failed!");

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("session init\n");
  // NNDEPLOY_LOGI("#. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("#. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("#. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = session_->init(tensor_repository_, op_repository_, is_dynamic_shape_,
                          max_shape_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "session init failed!");

  return status;
}

Net *createNet(ir::ModelDesc *model_desc, base::DeviceType device_type,
               base::PrecisionType precision_type) {
  Net *net = new Net();
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(net, "net is null!");
  net->setDeviceType(device_type);
  net->setPrecisionType(precision_type);
  net->setModelDesc(model_desc);
  return net;
}

}  // namespace net
}  // namespace nndeploy
