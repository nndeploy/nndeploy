#include "nndeploy/dag/edge/feedback_edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<FeedBackEdge>> g_feedback_edge_register(
    base::kEdgeTypeFeedback);

FeedBackEdge::FeedBackEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type) {
  data_packet_ = new DataPacket();
}

FeedBackEdge::~FeedBackEdge() { delete data_packet_; }

base::Status FeedBackEdge::setQueueMaxSize(int queue_max_size) {
  return base::kStatusCodeOk;
}

base::Status FeedBackEdge::construct() { return base::kStatusCodeOk; }

base::Status FeedBackEdge::set(device::Buffer *buffer, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(buffer, is_external);
}
device::Buffer *FeedBackEdge::create(device::Device *device,
                                     const device::BufferDesc &desc) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->create(device, desc);
}
bool FeedBackEdge::notifyWritten(device::Buffer *buffer) {
  return data_packet_->notifyWritten(buffer);
}
device::Buffer *FeedBackEdge::getBuffer(const Node *node) {
  return data_packet_->getBuffer();
}
device::Buffer *FeedBackEdge::getGraphOutputBuffer() {
  return data_packet_->getBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status FeedBackEdge::set(cv::Mat *cv_mat, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(cv_mat, is_external);
}
cv::Mat *FeedBackEdge::create(int rows, int cols, int type,
                              const cv::Vec3b &value) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->create(rows, cols, type, value);
}
bool FeedBackEdge::notifyWritten(cv::Mat *cv_mat) {
  return data_packet_->notifyWritten(cv_mat);
}
cv::Mat *FeedBackEdge::getCvMat(const Node *node) {
  return data_packet_->getCvMat();
}
cv::Mat *FeedBackEdge::getGraphOutputCvMat() {
  return data_packet_->getCvMat();
}
#endif

base::Status FeedBackEdge::set(device::Tensor *tensor, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(tensor, is_external);
}
device::Tensor *FeedBackEdge::create(device::Device *device,
                                     const device::TensorDesc &desc,
                                     const std::string &name) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->create(device, desc, name);
}
bool FeedBackEdge::notifyWritten(device::Tensor *tensor) {
  return data_packet_->notifyWritten(tensor);
}
device::Tensor *FeedBackEdge::getTensor(const Node *node) {
  return data_packet_->getTensor();
}
device::Tensor *FeedBackEdge::getGraphOutputTensor() {
  return data_packet_->getTensor();
}

base::Status FeedBackEdge::takeDataPacket(DataPacket *data_packet) {
  if (data_packet_ != nullptr) {
    delete data_packet_;
  }
  data_packet_ = data_packet;
  return base::kStatusCodeOk;
}
bool FeedBackEdge::notifyWritten(void *anything) {
  return data_packet_->notifyWritten(anything);
}
DataPacket *FeedBackEdge::getDataPacket(const Node *node) {
  return data_packet_;
}
DataPacket *FeedBackEdge::getGraphOutputDataPacket() { return data_packet_; }

base::Status FeedBackEdge::set(base::Param *param, bool is_external) {
  this->increaseIndex();
  data_packet_->setIndex(index_);
  return data_packet_->set(param, is_external);
}
bool FeedBackEdge::notifyWritten(base::Param *param) {
  return data_packet_->notifyWritten(param);
}

base::Param *FeedBackEdge::getParam(const Node *node) {
  auto *p = data_packet_->getParam();
  if (node && p) {
    last_read_index_[node] = data_packet_->getIndex();
  }
  return p;
}
base::Param *FeedBackEdge::getGraphOutputParam() {
  return data_packet_->getParam();
}

int64_t FeedBackEdge::getIndex(const Node *node) {
  return data_packet_->getIndex();
}
int64_t FeedBackEdge::getGraphOutputIndex() { return data_packet_->getIndex(); }

int FeedBackEdge::getPosition(const Node *node) { return 0; }
int FeedBackEdge::getGraphOutputPosition() { return 0; }

bool FeedBackEdge::hasBeenConsumedBy(const Node *n) {
  return last_read_index_.find(n) != last_read_index_.end();
}

base::EdgeUpdateFlag FeedBackEdge::update(const Node *node) {
  if (terminate_flag_) {
    return base::kEdgeUpdateFlagTerminate;
  }
  const int64_t cur = data_packet_->getIndex();  // -1 代表还没写
  if (cur < 0) {
    return base::kEdgeUpdateFlagTerminate;  // 还没有任何数据
  }
  int64_t last = -1;
  {
    auto it = last_read_index_.find(node);
    if (it != last_read_index_.end()) last = it->second;
  }

  if (cur == last) {
    // 对该消费者而言，没有比上次更新更“新”的数据
    return base::kEdgeUpdateFlagTerminate;
  }
  // 有比 last 更新的 token
  return base::kEdgeUpdateFlagComplete;
}

bool FeedBackEdge::requestTerminate() {
  terminate_flag_ = true;
  return true;
}

}  // namespace dag
}  // namespace nndeploy