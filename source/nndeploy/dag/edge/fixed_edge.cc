#include "nndeploy/dag/edge/fixed_edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<FixedEdge>> g_fixed_edge_register(
    kEdgeTypeFixed);

FixedEdge::FixedEdge(ParallelType paralle_type) : AbstractEdge(paralle_type) {
  data_packet_ = new DataPacket();
}

FixedEdge::~FixedEdge() { delete data_packet_; }

base::Status FixedEdge::construct() { return base::kStatusCodeOk; }

base::Status FixedEdge::set(device::Buffer *buffer, int index,
                            bool is_external) {
  return data_packet_->set(buffer, index, is_external);
}
base::Status FixedEdge::set(device::Buffer &buffer, int index) {
  return data_packet_->set(buffer, index);
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

base::Status FixedEdge::set(device::Mat *mat, int index, bool is_external) {
  return data_packet_->set(mat, index, is_external);
}
base::Status FixedEdge::set(device::Mat &mat, int index) {
  return data_packet_->set(mat, index);
}
device::Mat *FixedEdge::create(device::Device *device,
                               const device::MatDesc &desc, int index,
                               const std::string &name) {
  return data_packet_->create(device, desc, index, name);
}
bool FixedEdge::notifyWritten(device::Mat *mat) {
  return data_packet_->notifyWritten(mat);
}
device::Mat *FixedEdge::getMat(const Node *node) {
  return data_packet_->getMat();
}
device::Mat *FixedEdge::getGraphOutputMat() { return data_packet_->getMat(); }

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status FixedEdge::set(cv::Mat *cv_mat, int index, bool is_external) {
  return data_packet_->set(cv_mat, index, is_external);
}
base::Status FixedEdge::set(cv::Mat &cv_mat, int index) {
  return data_packet_->set(cv_mat, index);
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
base::Status FixedEdge::set(device::Tensor &tensor, int index) {
  return data_packet_->set(tensor, index);
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

base::Status FixedEdge::set(base::Param *param, int index, bool is_external) {
  return data_packet_->set(param, index, is_external);
}
base::Status FixedEdge::set(base::Param &param, int index) {
  return data_packet_->set(param, index);
}
base::Param *FixedEdge::getParam(const Node *node) {
  return data_packet_->getParam();
}
base::Param *FixedEdge::getGraphOutputParam() {
  return data_packet_->getParam();
}

base::Status FixedEdge::set(void *anything, int index, bool is_external) {
  return data_packet_->set(anything, index, is_external);
}
void *FixedEdge::getAnything(const Node *node) {
  return data_packet_->getAnything();
}
void *FixedEdge::getGraphOutputAnything() {
  return data_packet_->getAnything();
}

int FixedEdge::getIndex(const Node *node) { return data_packet_->getIndex(); }
int FixedEdge::getGraphOutputIndex() { return data_packet_->getIndex(); }

bool FixedEdge::updateData(const Node *node) { return true; }

bool FixedEdge::requestTerminate() { return true; }

}  // namespace dag
}  // namespace nndeploy