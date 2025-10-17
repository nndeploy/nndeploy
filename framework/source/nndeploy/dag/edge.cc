
#include "nndeploy/dag/edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

// Edge::Edge() : name_(""), abstact_edge_(nullptr) {}
// Edge::Edge(const std::string &name) : name_(name), abstact_edge_(nullptr) {}
Edge::Edge() {
  name_ = "edge_" + base::getUniqueString();
  abstact_edge_ = createEdge(base::kParallelTypeNone);
  if (abstact_edge_ == nullptr) {
    NNDEPLOY_LOGE("out of memory!\n");
    return;
  }
  // this->construct();
}
Edge::Edge(const std::string &name) : name_(name) {
  if (name.empty()) {
    name_ = "edge_" + base::getUniqueString();
  } else {
    name_ = name;
  }
  abstact_edge_ = createEdge(base::kParallelTypeNone);
  if (abstact_edge_ == nullptr) {
    NNDEPLOY_LOGE("out of memory!\n");
    return;
  }
  // this->construct();
}
Edge::~Edge() {
  // NNDEPLOY_LOGI("Edge[%s]::~Edge() START\n", name_.c_str());
  if (abstact_edge_ != nullptr) {
    delete abstact_edge_;
    abstact_edge_ = nullptr;
  }
  // NNDEPLOY_LOGI("Edge[%s]::~Edge() END\n", name_.c_str());
}

std::string Edge::getName() { return name_; }

base::Status Edge::setQueueMaxSize(int queue_max_size) {
  queue_max_size_ = queue_max_size;
  if (abstact_edge_ != nullptr) {
    abstact_edge_->setQueueMaxSize(queue_max_size_);
  }
  return base::kStatusCodeOk;
}
int Edge::getQueueMaxSize() { return queue_max_size_; }

base::Status Edge::setQueueOverflowPolicy(base::QueueOverflowPolicy policy,
                                          int drop_count) {
  queue_overflow_policy_ = policy;
  queue_drop_count_ = drop_count <= 0 ? 1 : drop_count;
  if (abstact_edge_ != nullptr) {
    abstact_edge_->setQueueOverflowPolicy(queue_overflow_policy_,
                                          queue_drop_count_);
  }
  return base::kStatusCodeOk;
}
base::QueueOverflowPolicy Edge::getQueueOverflowPolicy() {
  return queue_overflow_policy_;
}
int Edge::getQueueDropCount() { return queue_drop_count_; }

bool Edge::empty() {
  if (abstact_edge_ == nullptr) {
    return true;
  }
  return abstact_edge_->empty();
}

base::Status Edge::construct() { return abstact_edge_->construct(); }

base::Status Edge::set(device::Buffer *buffer, bool is_external) {
  this->setTypeInfo<device::Buffer>();
  return abstact_edge_->set(buffer, is_external);
}
base::Status Edge::set(device::Buffer &buffer) {
  this->setTypeInfo<device::Buffer>();
  return this->set(&buffer, true);
}
device::Buffer *Edge::create(device::Device *device,
                             const device::BufferDesc &desc) {
  this->setTypeInfo<device::Buffer>();
  return abstact_edge_->create(device, desc);
}
bool Edge::notifyWritten(device::Buffer *buffer) {
  return abstact_edge_->notifyWritten(buffer);
}
device::Buffer *Edge::getBuffer(const Node *node) {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (!type_info_->isType<device::Buffer>()) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getBuffer(node);
}
device::Buffer *Edge::getGraphOutputBuffer() {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (!type_info_->isType<device::Buffer>()) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getGraphOutputBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status Edge::set(cv::Mat *cv_mat, bool is_external) {
  this->setTypeInfo<cv::Mat>();
  return abstact_edge_->set(cv_mat, is_external);
}
base::Status Edge::set(cv::Mat &cv_mat) {
  this->setTypeInfo<cv::Mat>();
  return this->set(&cv_mat, true);
}
cv::Mat *Edge::create(int rows, int cols, int type, const cv::Vec3b &value) {
  this->setTypeInfo<cv::Mat>();
  return abstact_edge_->create(rows, cols, type, value);
}
bool Edge::notifyWritten(cv::Mat *cv_mat) {
  return abstact_edge_->notifyWritten(cv_mat);
}
cv::Mat *Edge::getCvMat(const Node *node) {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (!type_info_->isType<cv::Mat>()) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getCvMat(node);
}
cv::Mat *Edge::getGraphOutputCvMat() {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (!type_info_->isType<cv::Mat>()) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getGraphOutputCvMat();
}
#endif

base::Status Edge::set(device::Tensor *tensor, bool is_external) {
  this->setTypeInfo<device::Tensor>();
  return abstact_edge_->set(tensor, is_external);
}
base::Status Edge::set(device::Tensor &tensor) {
  this->setTypeInfo<device::Tensor>();
  return this->set(&tensor, true);
}
device::Tensor *Edge::create(device::Device *device,
                             const device::TensorDesc &desc,
                             std::string tensor_name) {
  this->setTypeInfo<device::Tensor>();
  if (tensor_name.empty()) {
    tensor_name = name_;
  }
  // if (tensor_name.empty()) {
  //   tensor_name = "tensor_" + base::getUniqueString();
  // }
  return abstact_edge_->create(device, desc, tensor_name);
}
bool Edge::notifyWritten(device::Tensor *tensor) {
  return abstact_edge_->notifyWritten(tensor);
}
device::Tensor *Edge::getTensor(const Node *node) {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (!type_info_->isType<device::Tensor>()) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getTensor(node);
}
device::Tensor *Edge::getGraphOutputTensor() {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (!type_info_->isType<device::Tensor>()) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getGraphOutputTensor();
}

base::Status Edge::set(base::Param *param, bool is_external) {
  this->setTypeInfo<base::Param>();
  return abstact_edge_->set(param, is_external);
}
base::Status Edge::set(base::Param &param) {
  this->setTypeInfo<base::Param>();
  return this->set(&param, true);
}
bool Edge::notifyWritten(base::Param *param) {
  return abstact_edge_->notifyWritten(param);
}
base::Param *Edge::getParam(const Node *node) {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (type_info_->getType() != EdgeTypeFlag::kParam) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getParam(node);
}
base::Param *Edge::getGraphOutputParam() {
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    std::unique_lock<std::mutex> lock(type_info_mutex_);
    type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
  }
  if (type_info_->getType() != EdgeTypeFlag::kParam) {
    // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
    return nullptr;
  }
  return abstact_edge_->getGraphOutputParam();
}

int64_t Edge::getIndex(const Node *node) {
  return abstact_edge_->getIndex(node);
}
int64_t Edge::getGraphOutputIndex() {
  return abstact_edge_->getGraphOutputIndex();
}
void Edge::resetIndex() { return abstact_edge_->resetIndex(); }
int Edge::getPosition(const Node *node) {
  return abstact_edge_->getPosition(node);
}
int Edge::getGraphOutputPosition() {
  return abstact_edge_->getGraphOutputPosition();
}

base::EdgeUpdateFlag Edge::update(const Node *node) {
  return abstact_edge_->update(node);
}

bool Edge::markGraphOutput() { return abstact_edge_->markGraphOutput(); }

base::Status Edge::setParallelType(const base::ParallelType &paralle_type) {
  if (abstact_edge_ == nullptr) {
    abstact_edge_ = createEdge(paralle_type);
    if (abstact_edge_ == nullptr) {
      NNDEPLOY_LOGE("out of memory!\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
  } else {
    abstact_edge_ = recreateEdge(abstact_edge_, paralle_type);
    if (abstact_edge_ == nullptr) {
      NNDEPLOY_LOGE("out of memory!\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
  }
  abstact_edge_->setQueueMaxSize(queue_max_size_);
  abstact_edge_->setQueueOverflowPolicy(queue_overflow_policy_,
                                        queue_drop_count_);
  return base::kStatusCodeOk;
}
base::ParallelType Edge::getParallelType() {
  return abstact_edge_->getParallelType();
}

base::Status Edge::increaseProducers(std::vector<Node *> &producers) {
  return abstact_edge_->increaseProducers(producers);
}
base::Status Edge::increaseConsumers(std::vector<Node *> &consumers) {
  return abstact_edge_->increaseConsumers(consumers);
}
std::vector<Node *> Edge::getProducers() {
  return abstact_edge_->getProducers();
}
std::vector<Node *> Edge::getConsumers() {
  if (abstact_edge_ == nullptr) {
    return std::vector<Node *>();
  }

  std::vector<Node *> consumers = abstact_edge_->getConsumers();
  std::vector<Node *> result;
  for (auto consumer : consumers) {
    if (consumer == nullptr) {
      // NNDEPLOY_LOGE("consumer is nullptr\n");
      continue;
    }
    result.push_back(consumer);
  }

  return result;
}

bool Edge::requestTerminate() { return abstact_edge_->requestTerminate(); }

base::Status Edge::setTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info) {
  type_info_ = type_info;
  if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
    type_info_cv_.notify_all();
  }
  return base::kStatusCodeOk;
}
std::shared_ptr<EdgeTypeInfo> Edge::getTypeInfo() { return type_info_; }
void Edge::setTypeName(const std::string &type_name) {
  if (type_info_ != nullptr) {
    type_info_->setTypeName(type_name);
  }
}
std::string Edge::getTypeName() { return type_info_->getTypeName(); }

bool Edge::checkTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info) {
  return *type_info_ == *type_info;
}

}  // namespace dag
}  // namespace nndeploy
