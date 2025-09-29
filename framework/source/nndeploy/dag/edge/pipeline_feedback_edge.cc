#include "nndeploy/dag/edge/pipeline_feedback_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineFeedbackEdge>>
    g_pipeline_feedback_edge_register(base::kEdgeTypePipelineFeedback);

PipelineFeedbackEdge::PipelineFeedbackEdge(base::ParallelType type)
    : AbstractEdge(type) {}

PipelineFeedbackEdge::~PipelineFeedbackEdge() {}

base::Status PipelineFeedbackEdge::setQueueMaxSize(int q) {
  if (q <= 0) q = 2;
  std::size_t value = round_up_pow2(static_cast<std::size_t>(q));
  if (ring_) {
    NNDEPLOY_LOGE("queue already constructed.\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  queue_max_size_ = value;
  return base::kStatusCodeOk;
}

base::Status PipelineFeedbackEdge::construct() {
  if (!ring_) ring_.reset(new Ring(queue_max_size_));
  consumer_size_ = static_cast<int>(consumers_.size());
  return base::kStatusCodeOk;
}

base::Status PipelineFeedbackEdge::set(device::Buffer *buffer,
                                       bool is_external) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);
  base::Status status = slot->set(buffer, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "Set data packet error.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return status;
}

device::Buffer *PipelineFeedbackEdge::create(device::Device *device,
                                             const device::BufferDesc &desc) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);

  device::Buffer *ret_value = slot->create(device, desc);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "Pipeline feedback edge create failed.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return nullptr;
  }

  return ret_value;
}

device::Buffer *PipelineFeedbackEdge::getBuffer(const Node *node) {
  Slot slot = nullptr;
  if (!ring_->pop(node, slot)) {
    NNDEPLOY_LOGE("Get data packet failed.\n");
    return nullptr;
  }
  return slot->getBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV

base::Status PipelineFeedbackEdge::set(cv::Mat *cv_mat, bool is_external) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);
  base::Status status = slot->set(cv_mat, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "Set data packet error.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return status;
}

cv::Mat *PipelineFeedbackEdge::create(int rows, int cols, int type,
                                      const cv::Vec3b &value) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);

  cv::Mat *ret_value = slot->create(rows, cols, type, value);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "Pipeline feedback edge create failed.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return nullptr;
  }

  return ret_value;
}

cv::Mat *PipelineFeedbackEdge::getCvMat(const Node *node) {
  Slot slot = nullptr;
  if (!ring_->pop(node, slot)) {
    NNDEPLOY_LOGE("Get data packet failed.\n");
    return nullptr;
  }
  return slot->getCvMat();
}

#endif

base::Status PipelineFeedbackEdge::set(device::Tensor *tensor,
                                       bool is_external) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);
  base::Status status = slot->set(tensor, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "Set data packet error.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return status;
}

device::Tensor *PipelineFeedbackEdge::create(device::Device *device,
                                             const device::TensorDesc &desc,
                                             const std::string &name) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);

  device::Tensor *ret_value = slot->create(device, desc, name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "Pipeline feedback edge create failed.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return nullptr;
  }

  return ret_value;
}

device::Tensor *PipelineFeedbackEdge::getTensor(const Node *node) {
  Slot slot = nullptr;
  if (!ring_->pop(node, slot)) {
    NNDEPLOY_LOGE("Get data packet failed.\n");
    return nullptr;
  }
  return slot->getTensor();
}

base::Status PipelineFeedbackEdge::set(base::Param *param, bool is_external) {
  Slot slot = std::make_shared<DataPacket>();
  this->increaseIndex();
  slot->setIndex(index_);
  base::Status status = slot->set(param, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "Set data packet error.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return status;
}

base::Param *PipelineFeedbackEdge::getParam(const Node *node) {
  Slot slot = nullptr;
  if (!ring_->pop(node, slot)) {
    NNDEPLOY_LOGE("Get data packet failed.\n");
    return nullptr;
  }
  return slot->getParam();
}

base::Status PipelineFeedbackEdge::takeDataPacket(DataPacket *data_packet) {
  Slot slot = std::make_shared<DataPacket>();
  base::Status status = slot->takeDataPacket(data_packet);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "Set data packet error.\n");
  if (!ring_->push(std::move(slot))) {
    NNDEPLOY_LOGE("Queue closed during push slot.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return status;
}

DataPacket *PipelineFeedbackEdge::getDataPacket(const Node *node) {
  Slot slot = nullptr;
  if (!ring_->pop(node, slot)) {
    NNDEPLOY_LOGE("Get data packet failed.\n");
    return nullptr;
  }
  return slot.get();
}

int64_t PipelineFeedbackEdge::getIndex(const Node *node) {
  return ring_->find_cid(node);
}
//=====================================================================
bool PipelineFeedbackEdge::notifyWritten(device::Buffer *buffer) {
  return true;
}
#ifdef ENABLE_NNDEPLOY_OPENCV
bool PipelineFeedbackEdge::notifyWritten(cv::Mat *cv_mat) { return true; }
#endif
bool PipelineFeedbackEdge::notifyWritten(device::Tensor *tensor) {
  return true;
}
bool PipelineFeedbackEdge::notifyWritten(base::Param *param) { return true; }
bool PipelineFeedbackEdge::notifyWritten(void *anything) { return true; }

device::Buffer *PipelineFeedbackEdge::getGraphOutputBuffer() { return nullptr; }
#ifdef ENABLE_NNDEPLOY_OPENCV
cv::Mat *PipelineFeedbackEdge::getGraphOutputCvMat() { return nullptr; }
#endif
device::Tensor *PipelineFeedbackEdge::getGraphOutputTensor() { return nullptr; }
base::Param *PipelineFeedbackEdge::getGraphOutputParam() { return nullptr; }

DataPacket *PipelineFeedbackEdge::getGraphOutputDataPacket() { return nullptr; }

int64_t PipelineFeedbackEdge::getGraphOutputIndex() { return 0; }

int PipelineFeedbackEdge::getPosition(const Node *node) { return 0; }

int PipelineFeedbackEdge::getGraphOutputPosition() {
  return getPosition(nullptr);
}

bool PipelineFeedbackEdge::requestTerminate() { return true; }

base::EdgeUpdateFlag PipelineFeedbackEdge::update(const Node *node) {
  return base::kEdgeUpdateFlagComplete;
}
//=====================================================================

}  // namespace dag
}  // namespace nndeploy