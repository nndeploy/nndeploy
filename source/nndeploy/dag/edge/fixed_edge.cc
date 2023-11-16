#include "nndeploy/dag/edge/fixed_edge.h"

namespace nndeploy {
namespace dag {

FixedEdge::FixedEdge(ParallelType paralle_type,
                     std::initializer_list<Node *> producers,
                     std::initializer_list<Node *> consumers)
    : AbstractEdge(paralle_type, producers, consumers) {
  data_packet_ = new DataPacket();
}

FixedEdge::~FixedEdge() { delete data_packet_; }

base::Status FixedEdge::set(device::Buffer *buffer, int index,
                            bool is_external) {
  return data_packet_->set(buffer, index, is_external);
}
base::Status FixedEdge::set(device::Buffer &buffer, int index,
                            bool is_external) {
  return data_packet_->set(buffer, index, is_external);
}
base::Status FixedEdge::create(device::Device *device,
                               const device::BufferDesc &desc, int index) {
  return data_packet_->create(device, desc, index);
}
device::Buffer *FixedEdge::getBuffer(const Node *comsumer) {
  return data_packet_->getBuffer();
}

base::Status FixedEdge::set(device::Mat *mat, int index, bool is_external) {
  return data_packet_->set(mat, index, is_external);
}
base::Status FixedEdge::set(device::Mat &mat, int index, bool is_external) {
  return data_packet_->set(mat, index, is_external);
}
base::Status FixedEdge::create(device::Device *device,
                               const device::MatDesc &desc, int index) {
  return data_packet_->create(device, desc, index);
}
device::Mat *FixedEdge::getMat(const Node *comsumer) {
  return data_packet_->getMat();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status FixedEdge::set(cv::Mat *cv_mat, int index, bool is_external) {
  return data_packet_->set(cv_mat, index, is_external);
}
base::Status FixedEdge::set(cv::Mat &cv_mat, int index, bool is_external) {
  return data_packet_->set(cv_mat, index, is_external);
}
cv::Mat *FixedEdge::getCvMat(const Node *comsumer) {
  return data_packet_->getCvMat();
}
#endif

base::Status FixedEdge::set(device::Tensor *tensor, int index,
                            bool is_external) {
  return data_packet_->set(tensor, index, is_external);
}
base::Status FixedEdge::set(device::Tensor &tensor, int index,
                            bool is_external) {
  return data_packet_->set(tensor, index, is_external);
}
base::Status FixedEdge::create(device::Device *device,
                               const device::TensorDesc &desc, int index) {
  return data_packet_->create(device, desc, index);
}
device::Tensor *FixedEdge::getTensor(const Node *comsumer) {
  return data_packet_->getTensor();
}

base::Status FixedEdge::set(base::Param *param, int index, bool is_external) {
  return data_packet_->set(param, index, is_external);
}
base::Status FixedEdge::set(base::Param &param, int index, bool is_external) {
  return data_packet_->set(param, index, is_external);
}
base::Param *FixedEdge::getParam(const Node *comsumer) {
  return data_packet_->getParam();
}

base::Status FixedEdge::set(void *anything, int index, bool is_external) {
  return data_packet_->set(anything, index, is_external);
}
void *FixedEdge::getAnything(const Node *comsumer) {
  return data_packet_->getAnything();
}

int FixedEdge::getIndex(const Node *comsumer) {
  return data_packet_->getIndex();
}

}  // namespace dag
}  // namespace nndeploy