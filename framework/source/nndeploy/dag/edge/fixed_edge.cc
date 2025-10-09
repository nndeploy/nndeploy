#include "nndeploy/dag/edge/fixed_edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<FixedEdge>> g_fixed_edge_register(
    base::kEdgeTypeFixed);

FixedEdge::FixedEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type) {
  data_packet_ = new DataPacket();
}

FixedEdge::~FixedEdge() { delete data_packet_; }

base::Status FixedEdge::setQueueMaxSize(int queue_max_size) {
  return base::kStatusCodeOk;
}

bool FixedEdge::empty() {
  if (data_packet_ == nullptr) {
    return true;
  }
  return data_packet_->empty();
}

base::Status FixedEdge::construct() { return base::kStatusCodeOk; }

base::Status FixedEdge::set(device::Buffer *buffer, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(buffer, is_external);
}
device::Buffer *FixedEdge::create(device::Device *device,
                                  const device::BufferDesc &desc) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->create(device, desc);
}
bool FixedEdge::notifyWritten(device::Buffer *buffer) {
  return data_packet_->notifyWritten(buffer);
}
device::Buffer *FixedEdge::getBuffer(const Node *node) {
  return data_packet_->getBuffer();
}
device::Buffer *FixedEdge::getGraphOutputBuffer() {
  return data_packet_->getBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status FixedEdge::set(cv::Mat *cv_mat, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(cv_mat, is_external);
}
cv::Mat *FixedEdge::create(int rows, int cols, int type,
                           const cv::Vec3b &value) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->create(rows, cols, type, value);
}
bool FixedEdge::notifyWritten(cv::Mat *cv_mat) {
  return data_packet_->notifyWritten(cv_mat);
}
cv::Mat *FixedEdge::getCvMat(const Node *node) {
  return data_packet_->getCvMat();
}
cv::Mat *FixedEdge::getGraphOutputCvMat() { return data_packet_->getCvMat(); }
#endif

base::Status FixedEdge::set(device::Tensor *tensor, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(tensor, is_external);
}
device::Tensor *FixedEdge::create(device::Device *device,
                                  const device::TensorDesc &desc,
                                  const std::string &name) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->create(device, desc, name);
}
bool FixedEdge::notifyWritten(device::Tensor *tensor) {
  return data_packet_->notifyWritten(tensor);
}
device::Tensor *FixedEdge::getTensor(const Node *node) {
  return data_packet_->getTensor();
}
device::Tensor *FixedEdge::getGraphOutputTensor() {
  return data_packet_->getTensor();
}

base::Status FixedEdge::takeDataPacket(DataPacket *data_packet) {
  if (data_packet_ != nullptr) {
    delete data_packet_;
  }
  data_packet_ = data_packet;
  return base::kStatusCodeOk;
}
bool FixedEdge::notifyWritten(void *anything) {
  return data_packet_->notifyWritten(anything);
}
DataPacket *FixedEdge::getDataPacket(const Node *node) { return data_packet_; }
DataPacket *FixedEdge::getGraphOutputDataPacket() { return data_packet_; }

base::Status FixedEdge::set(base::Param *param, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(param, is_external);
}
bool FixedEdge::notifyWritten(base::Param *param) {
  return data_packet_->notifyWritten(param);
}
base::Param *FixedEdge::getParam(const Node *node) {
  return data_packet_->getParam();
}
base::Param *FixedEdge::getGraphOutputParam() {
  return data_packet_->getParam();
}

int64_t FixedEdge::getIndex(const Node *node) {
  return data_packet_->getIndex();
}
int64_t FixedEdge::getGraphOutputIndex() { return data_packet_->getIndex(); }

int FixedEdge::getPosition(const Node *node) { return 0; }
int FixedEdge::getGraphOutputPosition() { return 0; }

base::EdgeUpdateFlag FixedEdge::update(const Node *node) {
  if (terminate_flag_) {
    return base::kEdgeUpdateFlagTerminate;
  } else {
    return base::kEdgeUpdateFlagComplete;
  }
}

bool FixedEdge::requestTerminate() {
  terminate_flag_ = true;
  return true;
}

}  // namespace dag
}  // namespace nndeploy