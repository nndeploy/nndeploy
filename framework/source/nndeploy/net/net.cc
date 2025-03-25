
#include "nndeploy/net/net.h"

#include "nndeploy/net/optimizer.h"
#include "nndeploy/net/runtime.h"
#include "nndeploy/net/runtime/pipeline_runtime.h"
#include "nndeploy/net/runtime/sequential_runtime.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace net {

Net::Net() : op::Op() {}

Net::~Net() {}

base::Status Net::setInterpret(ir::Interpret *interpret) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(interpret, "interpret is null!");
  model_desc_ = interpret->getModelDesc();
  return status;
}

base::Status Net::setModelDesc(ir::ModelDesc *model_desc) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(model_desc, "model_desc is null!");
  model_desc_ = model_desc;
  return status;
}

base::Status Net::setDynamicShape(bool is_dynamic_shape,
                                  base::ShapeMap &min_shape,
                                  base::ShapeMap &opt_shape,
                                  base::ShapeMap &max_shape) {
  base::Status status = base::kStatusCodeOk;
  is_dynamic_shape_ = is_dynamic_shape;
  min_shape_ = min_shape;
  opt_shape_ = opt_shape;
  max_shape_ = max_shape;
  return status;
}

base::Status Net::setTensorPoolType(TensorPoolType tensor_pool_type) {
  base::Status status = base::kStatusCodeOk;
  tensor_pool_type_ = tensor_pool_type;
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
    NNDEPLOY_LOGE("Failed to create Op[name:%s, type:%s, device_type:%s]\n",
                  name.c_str(), ir::opTypeToString(op_type).c_str(),
                  base::deviceTypeToString(device_type).c_str());
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

  status = inferDataType();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "inferDataType failed!");

  status = inferShape();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "inferShape failed!");

  status = inferDataFormat();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "inferDataFormat failed!");

  // 即使是设备相关的图优化，也可以放在优化器中做
  // 经过这一次图优化之后
  if (net_opt_flag_) {
    status = optimizer();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "graph optimizer failed!");
  }

  status = this->runtime();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "graph runtime failed!");

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
  status = runtime_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime deinit failed!");
  delete runtime_;
  runtime_ = nullptr;

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

  enable_pass_.clear();
  disable_pass_.clear();

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
/*
 * 静态输入shape：
 * # input的shape不为空
 * # input的shape为空，opt_shape_中存在该input的shape
 * 动态shape：
 * # is_dynamic_shape_位true，max_shape_中存在该input的max shape
 * # is_dynamic_shape_位true，max_shape_中不存在该input的max shape
 * ## 大语言模型
 * ## 无法得知input的shape的cv模型
 */
base::Status Net::inferShape() {
  base::Status status = base::kStatusCodeOk;
  bool is_infer_shape = true;
  for (auto input : inputs_) {
    if (input->getShape().empty()) {
      is_infer_shape = false;
      break;
    }
  }
  if (is_infer_shape) {
    for (auto iter : op_repository_) {
      // NNDEPLOY_LOGI("Op inferShape: %s\n", iter->op_->getName().c_str());
      status = iter->op_->inferShape();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Op inferShape failed: %s\n",
                      iter->op_->getName().c_str());
        return status;
      }
      // auto output = iter->op_->getOutput();
      // output->print();
    }
  }
  return status;
};
base::Status Net::inferDataFormat() {
  base::Status status = base::kStatusCodeOk;
  auto device = device::getDevice(device_type_);
  bool is_infer_data_format = true;
  for (auto input : inputs_) {
    if (input->getShape().empty()) {
      is_infer_data_format = false;
      break;
    } else {
      base::DataFormat data_format =
          device->getDataFormatByShape(input->getShape());
      input->setDataFormat(data_format);
    }
  }
  if (is_infer_data_format) {
    for (auto iter : op_repository_) {
      status = iter->op_->inferDataFormat();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "inferDataFormat failed!");
    }
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
  status = runtime_->reshape(shape_map);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime preRun failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(false);

  return status;
};

int64_t Net::getMemorySize() { return runtime_->getMemorySize(); }
base::Status Net::setMemory(device::Buffer *buffer) {
  return runtime_->setMemory(buffer);
}

base::Status Net::preRun() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = runtime_->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime preRun failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setRunningFlag(false);

  return status;
};

uint64_t Net::getFlops() {
  uint64_t flops = 0;
  for (auto iter : op_repository_) {
    flops += iter->op_->getFlops();
  }
  return flops;
}

base::Status Net::run() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(true);

  // 输入
#if 0
  for (size_t i = 0; i < inputs_.size(); ++i) {
    std::string path = "./net/";
    std::string name = inputs_[i]->getName();
    std::string filename = name;
    size_t pos = 0;
    while ((pos = filename.find('/')) != std::string::npos) {
      filename.replace(pos, 1, "_");
    }
    filename = path + filename + ".csv";
    std::ofstream output_file(filename, std::ios::trunc);
    if (output_file.is_open()) {
      inputs_[i]->print(output_file);
      output_file.close();
    } else {
      NNDEPLOY_LOGE("无法打开文件：%s", filename.c_str());
    }
  }
#endif

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Op run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = runtime_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime run failed!");

  // 输出
#if 0
  device::Device *device = device::getDevice(device_type_);
  stream_->synchronize();
  for (size_t i = 0; i < outputs_.size(); ++i) {
    std::string path = "./net/";
    std::string name = outputs_[i]->getName();
    std::string filename = name;
    size_t pos = 0;
    while ((pos = filename.find('/')) != std::string::npos) {
      filename.replace(pos, 1, "_");
    }
    filename = path + filename + ".csv";
    std::ofstream output_file(filename, std::ios::trunc);
    if (output_file.is_open()) {
      outputs_[i]->print(output_file);
      output_file.close();
    } else {
      NNDEPLOY_LOGE("无法打开文件：%s", filename.c_str());
    }
  }
#endif

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
  status = runtime_->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime run failed!");

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
      // insertUnique(input_names, input_name);
      input_names.push_back(input_name);
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
    input_wrapper->input_output_type_ = kInput;
    inputs_.emplace_back(input_wrapper->tensor_);

    input_wrapper->tensor_->setDataType(input->data_type_);

    /*
     * 静态输入shape：
     * # input的shape不为空
     * # input的shape为空，opt_shape_中存在该input的shape
     * 动态shape：
     * # is_dynamic_shape_位true，max_shape_中存在该input的max shape
     * # is_dynamic_shape_位true，max_shape_中不存在该input的max shape
     * ## 大语言模型
     * ## 无法得知input的shape的cv模型
     */
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
    output_wrapper->input_output_type_ = kOutput;
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
    // NNDEPLOY_LOGI("tensor name = %s\n", tensor_wrapper->name_.c_str());
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(tensor_wrapper->tensor_,
                                         "tensor_repository_ tensor is null!");
    if (tensor_wrapper->producers_.empty() &&
        tensor_wrapper->consumers_.empty()) {
      NNDEPLOY_LOGI("this tensor[%s] is useless!\n",
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
  // NNDEPLOY_LOGI("set stream\n");
  // NNDEPLOY_LOGI("##############\n");
  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
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
  std::unique_ptr<net::Optimizer> optimizer =
      std::make_unique<net::Optimizer>();
  status = optimizer->init(device_type_, enable_pass_, disable_pass_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "optimizer init failed!");
  status = optimizer->optimize(tensor_repository_, op_repository_, this);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "optimizer optimize failed!");
  return status;
}

base::Status Net::runtime() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("create runtime\n");
  // NNDEPLOY_LOGI("##############\n");
  runtime_ = createRuntime(device_type_, parallel_type_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(runtime_, "Create runtime failed!");
  // if (parallel_type_ == base::kParallelTypePipeline) {
  //   for (auto input : inputs_) {
  //     PipelineTensor *pipeline_input = new PipelineTensor();
  //     pipeline_input_output_tensors_[input] = pipeline_input;
  //   }
  //   for (auto output : outputs_) {
  //     PipelineTensor *pipeline_output = new PipelineTensor();
  //     // 外部消费者
  //     pipeline_output->consumers_.emplace_back(nullptr);
  //     pipeline_output->current_index_[nullptr] = 0;
  //     pipeline_input_output_tensors_[output] = pipeline_output;
  //   }
  //   PipelineRuntime *pipeline_runtime =
  //       dynamic_cast<PipelineRuntime *>(runtime_);
  //   pipeline_runtime->setInputOutputTensors(pipeline_input_output_tensors_);
  //   pipeline_runtime->setWorkers(worker_num_, device_types_);
  // }

  runtime_->setWorkers(worker_num_, device_types_);

  // 设置流
  runtime_->setStream(stream_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "runtime setStream failed!");

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("runtime init\n");
  // NNDEPLOY_LOGI("#. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("#. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("#. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = runtime_->init(tensor_repository_, op_repository_, inputs_, outputs_,
                          is_dynamic_shape_, max_shape_, tensor_pool_type_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "runtime init failed!");

  return status;
}

base::Status Net::enableOpt(bool flag) {
  net_opt_flag_ = flag;
  return base::kStatusCodeOk;
}

base::Status Net::setEnablePass(std::set<OptPassType> enable_pass) {
  enable_pass_ = enable_pass;
  return base::kStatusCodeOk;
}

base::Status Net::setDisablePass(std::set<OptPassType> disable_pass) {
  disable_pass_ = disable_pass;
  return base::kStatusCodeOk;
}

base::Status Net::setWorkers(int worker_num,
                             std::vector<base::DeviceType> device_types) {
  worker_num_ = worker_num;
  device_types_ = device_types;
  return base::kStatusCodeOk;
}

base::Status Net::copyToInputTensor(device::Tensor *tensor) {
  return runtime_->copyToInputTensor(tensor);
  // device::Tensor *src_tensor = tensor;
  // device::Tensor *dst_tensor = nullptr;
  // runtime_->allocateInput();
  // for (auto input : inputs_) {
  //   if (input->getName() == src_tensor->getName()) {
  //     dst_tensor = input;
  //     break;
  //   }
  // }
  // if (dst_tensor == nullptr) {
  //   NNDEPLOY_LOGE("copyToInputTensor failed! input tensor not found!\n");
  //   return base::kStatusCodeErrorInvalidValue;
  // }
  // if (parallel_type_ == base::kParallelTypeSequential ||
  //     parallel_type_ == base::kParallelTypeNone) {
  //   // src_tensor->getDesc().print();
  //   // dst_tensor->getDesc().print();
  //   if (src_tensor->getData() != dst_tensor->getData()) {
  //     base::Status status = src_tensor->copyTo(dst_tensor);
  //     NNDEPLOY_RETURN_ON_NEQ(
  //         status, base::kStatusCodeOk,
  //         "copy external_input_tensor to internal_input_tensor failed!");
  //   }
  // } else if (parallel_type_ == base::kParallelTypePipeline) {
  //   auto iter = pipeline_input_output_tensors_.find(dst_tensor);
  //   if (iter == pipeline_input_output_tensors_.end()) {
  //     NNDEPLOY_LOGE("pipeline_input_output_tensors_ not found
  //     dst_tensor!\n"); return base::kStatusCodeErrorInvalidValue;
  //   }
  //   PipelineTensor *pipeline_tensor = iter->second;
  //   device::Tensor *vec_dst_tensor = new device::Tensor(*src_tensor);
  //   static int count = 0;
  //   if (count == 0) {
  //     std::string filename = "vec_dst_tensor" + src_tensor->getName() +
  //     ".csv"; size_t pos = 0; while ((pos = filename.find('/')) !=
  //     std::string::npos) {
  //       filename.replace(pos, 1, "_");
  //     }
  //     std::ofstream input_file(filename, std::ios::trunc);
  //     if (input_file.is_open()) {
  //       vec_dst_tensor->print(input_file);
  //       input_file.close();
  //     } else {
  //       NNDEPLOY_LOGE("can't open file: %s", filename.c_str());
  //     }
  //   }
  //   count++;
  //   pipeline_tensor->push(vec_dst_tensor);
  // } else {
  //   NNDEPLOY_LOGE("parallel_type is not supported!\n");
  //   return base::kStatusCodeErrorInvalidValue;
  // }
  // return base::kStatusCodeOk;
}

device::Tensor *Net::getOutputTensorAfterRun(const std::string &name,
                                             base::DeviceType device_type,
                                             bool is_copy,
                                             base::DataFormat data_format) {
  return runtime_->getOutputTensorAfterRun(name, device_type, is_copy,
                                           data_format);
  // device::Device *device = device::getDevice(device_type);
  // device::Tensor *internal_output_tensor = nullptr;
  // for (auto output : outputs_) {
  //   if (output->getName() == name) {
  //     internal_output_tensor = output;
  //     break;
  //   }
  // }
  // if (internal_output_tensor == nullptr) {
  //   NNDEPLOY_LOGE("Not exsit output[%s].\n", name.c_str());
  //   return nullptr;
  // }
  // if (parallel_type_ == base::kParallelTypeSequential ||
  //     parallel_type_ == base::kParallelTypeNone) {
  //   bool flag = is_copy || (internal_output_tensor->getDevice() != device);
  //   device::Tensor *output_tensor = nullptr;
  //   if (flag) {
  //     output_tensor =
  //         new device::Tensor(device, internal_output_tensor->getDesc(),
  //         name);
  //     // internal_tensor->getDesc().print();
  //     // output_tensor->getDesc().print();
  //     internal_output_tensor->copyTo(output_tensor);
  //     return output_tensor;
  //   } else {
  //     return internal_output_tensor;
  //   }
  // } else if (parallel_type_ == base::kParallelTypePipeline) {
  //   auto iter = pipeline_input_output_tensors_.find(internal_output_tensor);
  //   if (iter == pipeline_input_output_tensors_.end()) {
  //     NNDEPLOY_LOGE(
  //         "pipeline_input_output_tensors_ not found
  //         internal_output_tensor!\n");
  //     return nullptr;
  //   }
  //   PipelineTensor *pipeline_internal_output_tensor = iter->second;
  //   NNDEPLOY_LOGI("pipeline_internal_output_tensor->tensors_.size() %d\n",
  //                 pipeline_internal_output_tensor->tensors_.size());
  //   device::Tensor *pipeline_output_tensor =
  //       pipeline_internal_output_tensor->pop(nullptr);
  //   if (pipeline_output_tensor == nullptr) {
  //     NNDEPLOY_LOGE("pipeline_output_tensor is nullptr!\n");
  //     return nullptr;
  //   }
  //   NNDEPLOY_LOGI("pipeline_output_tensor NAME %s\n",
  //                 pipeline_output_tensor->getName().c_str());
  //   bool flag = is_copy || (pipeline_output_tensor->getDevice() != device);
  //   device::Tensor *output_tensor = nullptr;
  //   if (flag) {
  //     output_tensor =
  //         new device::Tensor(device, pipeline_output_tensor->getDesc(),
  //         name);
  //     pipeline_output_tensor->copyTo(output_tensor);
  //     return output_tensor;
  //   } else {
  //     return pipeline_output_tensor;
  //   }
  // } else {
  //   NNDEPLOY_LOGE("parallel_type is not supported!\n");
  //   return nullptr;
  // }
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
