
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
  NNDEPLOY_LOGI("Edge[%s]::~Edge() START\n", name_.c_str());
  if (abstact_edge_ != nullptr) {
    delete abstact_edge_;
    abstact_edge_ = nullptr;
  }
  NNDEPLOY_LOGI("Edge[%s]::~Edge() END\n", name_.c_str());
}

std::string Edge::getName() { return name_; }

base::Status Edge::construct() { return abstact_edge_->construct(); }

base::Status Edge::set(device::Buffer *buffer, int index, bool is_external) {
  return abstact_edge_->set(buffer, index, is_external);
}
base::Status Edge::set(device::Buffer &buffer, int index) {
  return this->set(&buffer, index, true);
}
device::Buffer *Edge::create(device::Device *device,
                             const device::BufferDesc &desc, int index) {
  return abstact_edge_->create(device, desc, index);
}
bool Edge::notifyWritten(device::Buffer *buffer) {
  return abstact_edge_->notifyWritten(buffer);
}
device::Buffer *Edge::getBuffer(const Node *node) {
  return abstact_edge_->getBuffer(node);
}
device::Buffer *Edge::getGraphOutputBuffer() {
  return abstact_edge_->getGraphOutputBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status Edge::set(cv::Mat *cv_mat, int index, bool is_external) {
  return abstact_edge_->set(cv_mat, index, is_external);
}
base::Status Edge::set(cv::Mat &cv_mat, int index) {
  return this->set(&cv_mat, index, true);
}
cv::Mat *Edge::create(int rows, int cols, int type, const cv::Vec3b &value,
                      int index) {
  return abstact_edge_->create(rows, cols, type, value, index);
}
bool Edge::notifyWritten(cv::Mat *cv_mat) {
  return abstact_edge_->notifyWritten(cv_mat);
}
cv::Mat *Edge::getCvMat(const Node *node) {
  return abstact_edge_->getCvMat(node);
}
cv::Mat *Edge::getGraphOutputCvMat() {
  return abstact_edge_->getGraphOutputCvMat();
}
#endif

base::Status Edge::set(device::Tensor *tensor, int index, bool is_external) {
  return abstact_edge_->set(tensor, index, is_external);
}
base::Status Edge::set(device::Tensor &tensor, int index) {
  return this->set(&tensor, index, true);
}
device::Tensor *Edge::create(device::Device *device,
                             const device::TensorDesc &desc, int index,
                             std::string tensor_name) {
  if (tensor_name.empty()) {
    tensor_name = name_;
  }
  // if (tensor_name.empty()) {
  //   tensor_name = "tensor_" + base::getUniqueString();
  // }
  return abstact_edge_->create(device, desc, index, tensor_name);
}
bool Edge::notifyWritten(device::Tensor *tensor) {
  return abstact_edge_->notifyWritten(tensor);
}
device::Tensor *Edge::getTensor(const Node *node) {
  return abstact_edge_->getTensor(node);
}
device::Tensor *Edge::getGraphOutputTensor() {
  return abstact_edge_->getGraphOutputTensor();
}

base::Status Edge::set(base::Param *param, int index, bool is_external) {
  return abstact_edge_->set(param, index, is_external);
}
base::Status Edge::set(base::Param &param, int index) {
  return this->set(&param, index, true);
}
bool Edge::notifyWritten(base::Param *param) {
  return abstact_edge_->notifyWritten(param);
}
base::Param *Edge::getParam(const Node *node) {
  return abstact_edge_->getParam(node);
}
base::Param *Edge::getGraphOutputParam() {
  return abstact_edge_->getGraphOutputParam();
}

int Edge::getIndex(const Node *node) { return abstact_edge_->getIndex(node); }
int Edge::getGraphOutputIndex() { return abstact_edge_->getGraphOutputIndex(); }

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

bool Edge::requestTerminate() { return abstact_edge_->requestTerminate(); }

}  // namespace dag
}  // namespace nndeploy
