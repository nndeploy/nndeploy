
#include "nndeploy/dag/edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

Edge::Edge() : name_(""), abstact_edge_(nullptr) {}
Edge::Edge(const std::string &name) : name_(name), abstact_edge_(nullptr) {}
Edge::~Edge() { delete abstact_edge_; }

std::string Edge::getName() { return name_; }

base::Status Edge::construct() { return abstact_edge_->construct(); }

base::Status Edge::set(device::Buffer *buffer, int index, bool is_external) {
  return abstact_edge_->set(buffer, index, is_external);
}
base::Status Edge::set(device::Buffer &buffer, int index) {
  return abstact_edge_->set(buffer, index);
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

base::Status Edge::set(device::Mat *mat, int index, bool is_external) {
  return abstact_edge_->set(mat, index, is_external);
}
base::Status Edge::set(device::Mat &mat, int index) {
  return abstact_edge_->set(mat, index);
}
device::Mat *Edge::create(device::Device *device, const device::MatDesc &desc,
                          int index) {
  return abstact_edge_->create(device, desc, index, name_);
}
bool Edge::notifyWritten(device::Mat *mat) {
  return abstact_edge_->notifyWritten(mat);
}
device::Mat *Edge::getMat(const Node *node) {
  return abstact_edge_->getMat(node);
}
device::Mat *Edge::getGraphOutputMat() {
  return abstact_edge_->getGraphOutputMat();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status Edge::set(cv::Mat *cv_mat, int index, bool is_external) {
  return abstact_edge_->set(cv_mat, index, is_external);
}
base::Status Edge::set(cv::Mat &cv_mat, int index) {
  return abstact_edge_->set(cv_mat, index);
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
  return abstact_edge_->set(tensor, index);
}
device::Tensor *Edge::create(device::Device *device,
                             const device::TensorDesc &desc, int index) {
  return abstact_edge_->create(device, desc, index, name_);
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
  return abstact_edge_->set(param, index);
}
base::Param *Edge::getParam(const Node *node) {
  return abstact_edge_->getParam(node);
}
base::Param *Edge::getGraphOutputParam() {
  return abstact_edge_->getGraphOutputParam();
}

base::Status Edge::set(void *anything, int index, bool is_external) {
  return abstact_edge_->set(anything, index, is_external);
}
void *Edge::getAnything(const Node *node) {
  return abstact_edge_->getAnything(node);
}
void *Edge::getGraphOutputAnything() {
  return abstact_edge_->getGraphOutputAnything();
}

int Edge::getIndex(const Node *node) { return abstact_edge_->getIndex(node); }
int Edge::getGraphOutputIndex() { return abstact_edge_->getGraphOutputIndex(); }

int Edge::getPosition(const Node *node) {
  return abstact_edge_->getPosition(node);
}
int Edge::getGraphOutputPosition() {
  return abstact_edge_->getGraphOutputPosition();
}

EdgeUpdateFlag Edge::update(const Node *node) {
  return abstact_edge_->update(node);
}

bool Edge::markGraphOutput() { return abstact_edge_->markGraphOutput(); }

base::Status Edge::setParallelType(const ParallelType &paralle_type) {
  if (abstact_edge_ == nullptr) {
    abstact_edge_ = createEdge(paralle_type);
    if (abstact_edge_ == nullptr) {
      NNDEPLOY_LOGE("out of memory!\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
  } else {
    ParallelType cur_paralle_type = abstact_edge_->getParallelType();
    if ((int)paralle_type > (int)cur_paralle_type) {
      AbstractEdge *new_abstact_edge = createEdge(paralle_type);
      if (new_abstact_edge == nullptr) {
        NNDEPLOY_LOGE("out of memory!\n");
        return base::kStatusCodeErrorOutOfMemory;
      }
      std::vector<Node *> producers = abstact_edge_->getProducers();
      new_abstact_edge->increaseProducers(producers);
      std::vector<Node *> consumers = abstact_edge_->getConsumers();
      new_abstact_edge->increaseConsumers(consumers);
      delete abstact_edge_;
      abstact_edge_ = new_abstact_edge;
    }
  }
  return base::kStatusCodeOk;
}
ParallelType Edge::getParallelType() {
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
