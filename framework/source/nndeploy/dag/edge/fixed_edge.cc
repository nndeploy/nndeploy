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

base::Status FixedEdge::construct() { return base::kStatusCodeOk; }

base::Status FixedEdge::set(device::Buffer *buffer, int index,
                            bool is_external) {
  return data_packet_->set(buffer, index, is_external);
}
device::Buffer *FixedEdge::create(device::Device *device,
                                  const device::BufferDesc &desc, int index) {
  return data_packet_->create(device, desc, index);
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
base::Status FixedEdge::set(cv::Mat *cv_mat, int index, bool is_external) {
  return data_packet_->set(cv_mat, index, is_external);
}
cv::Mat *FixedEdge::create(int rows, int cols, int type, const cv::Vec3b& value,
                           int index) {
  return data_packet_->create(rows, cols, type, value, index);
}
bool FixedEdge::notifyWritten(cv::Mat *cv_mat) {
  return data_packet_->notifyWritten(cv_mat);
}
cv::Mat *FixedEdge::getCvMat(const Node *node) {
  return data_packet_->getCvMat();
}
cv::Mat *FixedEdge::getGraphOutputCvMat() { return data_packet_->getCvMat(); }
#endif

base::Status FixedEdge::set(device::Tensor *tensor, int index,
                            bool is_external) {
  return data_packet_->set(tensor, index, is_external);
}
device::Tensor *FixedEdge::create(device::Device *device,
                                  const device::TensorDesc &desc, int index,
                                  const std::string &name) {
  return data_packet_->create(device, desc, index, name);
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
  data_packet_ = data_packet;
  data_packet_ = nullptr;
  return base::kStatusCodeOk;
}
bool FixedEdge::notifyAnyWritten(void *anything) {
  return data_packet_->notifyAnyWritten(anything);
}
DataPacket *FixedEdge::getDataPacket(const Node *node) {
  return data_packet_;
}
DataPacket *FixedEdge::getGraphOutputDataPacket() {
  return data_packet_;
}

base::Status FixedEdge::set(base::Param *param, int index, bool is_external) {
  return data_packet_->set(param, index, is_external);
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

int FixedEdge::getIndex(const Node *node) { return data_packet_->getIndex(); }
int FixedEdge::getGraphOutputIndex() { return data_packet_->getIndex(); }

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