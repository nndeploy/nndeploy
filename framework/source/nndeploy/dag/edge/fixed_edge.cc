
#include "nndeploy/dag/edge/fixed_edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<FixedEdge>> g_fixed_edge_register(
    base::kEdgeTypeFixed);

FixedEdge::FixedEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type) {
  data_packet_ = std::make_shared<DataPacket>();
}

FixedEdge::~FixedEdge() = default;

base::Status FixedEdge::setQueueMaxSize(int queue_max_size) {
  return base::kStatusCodeOk;
}

bool FixedEdge::empty() {
  return acquirePacket(nullptr, false)->empty();
}

base::Status FixedEdge::construct() {
  std::lock_guard<std::mutex> lock(mutex_);
  terminate_flag_ = false;
  return base::kStatusCodeOk;
}

base::Status FixedEdge::set(device::Buffer *buffer, bool is_external) {
  auto packet = std::make_unique<DataPacket>();
  packet->set(buffer, is_external);
  return takeDataPacket(packet.release());
}
device::Buffer *FixedEdge::create(device::Device *device,
                                  const device::BufferDesc &desc) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *current = data_packet_->getBuffer();
    if (data_packet_.use_count() == 1 && data_packet_->ownsPayload() && current &&
        current->getDevice() == device && current->getDesc() == desc) {
      increaseIndex();
      data_packet_->setIndex(index_);
      return data_packet_->create(device, desc);
    }
  }
  auto packet = std::make_unique<DataPacket>();
  auto *result = packet->create(device, desc);
  takeDataPacket(packet.release());
  return result;
}
bool FixedEdge::notifyWritten(device::Buffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_packet_->notifyWritten(buffer);
}
device::Buffer *FixedEdge::getBuffer(const Node *node) {
  return acquirePacket(nullptr, false)->getBuffer();
}
device::Buffer *FixedEdge::getGraphOutputBuffer() {
  return acquirePacket(nullptr, false)->getBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status FixedEdge::set(cv::Mat *cv_mat, bool is_external) {
  auto packet = std::make_unique<DataPacket>();
  packet->set(cv_mat, is_external);
  return takeDataPacket(packet.release());
}
cv::Mat *FixedEdge::create(int rows, int cols, int type,
                           const cv::Vec3b &value) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *current = data_packet_->getCvMat();
    if (data_packet_.use_count() == 1 && data_packet_->ownsPayload() && current &&
        current->rows == rows && current->cols == cols && current->type() == type) {
      increaseIndex();
      data_packet_->setIndex(index_);
      return data_packet_->create(rows, cols, type, value);
    }
  }
  auto packet = std::make_unique<DataPacket>();
  auto *result = packet->create(rows, cols, type, value);
  takeDataPacket(packet.release());
  return result;
}
bool FixedEdge::notifyWritten(cv::Mat *cv_mat) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_packet_->notifyWritten(cv_mat);
}
cv::Mat *FixedEdge::getCvMat(const Node *node) {
  return acquirePacket(nullptr, false)->getCvMat();
}
cv::Mat *FixedEdge::getGraphOutputCvMat() { return acquirePacket(nullptr, false)->getCvMat(); }
#endif

base::Status FixedEdge::set(device::Tensor *tensor, bool is_external) {
  auto packet = std::make_unique<DataPacket>();
  packet->set(tensor, is_external);
  return takeDataPacket(packet.release());
}
device::Tensor *FixedEdge::create(device::Device *device,
                                  const device::TensorDesc &desc,
                                  const std::string &name) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *current = data_packet_->getTensor();
    if (data_packet_.use_count() == 1 && data_packet_->ownsPayload() && current &&
        current->getDevice() == device && current->getDesc() == desc) {
      increaseIndex();
      data_packet_->setIndex(index_);
      return data_packet_->create(device, desc, name);
    }
  }
  auto packet = std::make_unique<DataPacket>();
  auto *result = packet->create(device, desc, name);
  takeDataPacket(packet.release());
  return result;
}
bool FixedEdge::notifyWritten(device::Tensor *tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_packet_->notifyWritten(tensor);
}
device::Tensor *FixedEdge::getTensor(const Node *node) {
  return acquirePacket(nullptr, false)->getTensor();
}
device::Tensor *FixedEdge::getGraphOutputTensor() {
  return acquirePacket(nullptr, false)->getTensor();
}

base::Status FixedEdge::takeDataPacket(DataPacket *data_packet) {
  if (!data_packet) return base::kStatusCodeErrorInvalidParam;
  std::shared_ptr<DataPacket> incoming(data_packet);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    incoming->retainSamePayload(data_packet_);
    increaseIndex();
    incoming->setIndex(index_);
    data_packet_.swap(incoming);
  }
  return base::kStatusCodeOk;
}
bool FixedEdge::notifyWritten(void *anything) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_packet_->notifyWritten(anything);
}
std::shared_ptr<DataPacket> FixedEdge::acquirePacket(const Node *, bool, EdgeTypeFlag expected,
                                                     const std::type_info *type) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_packet_->matches(expected, type) ? data_packet_ : nullptr;
}
DataPacket *FixedEdge::getDataPacket(const Node *node) { return acquirePacket(nullptr, false).get(); }
DataPacket *FixedEdge::getGraphOutputDataPacket() { return acquirePacket(nullptr, false).get(); }

base::Status FixedEdge::set(base::Param *param, bool is_external) {
  auto packet = std::make_unique<DataPacket>();
  packet->set(param, is_external);
  return takeDataPacket(packet.release());
}
bool FixedEdge::notifyWritten(base::Param *param) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_packet_->notifyWritten(param);
}
base::Param *FixedEdge::getParam(const Node *node) {
  return acquirePacket(nullptr, false)->getParam();
}
base::Param *FixedEdge::getGraphOutputParam() {
  return acquirePacket(nullptr, false)->getParam();
}
void *FixedEdge::getGraphOutputPtr() { return acquirePacket(nullptr, false)->getDataPtr(); }

int64_t FixedEdge::getIndex(const Node *node) {
  return acquirePacket(nullptr, false)->getIndex();
}
int64_t FixedEdge::getGraphOutputIndex() { return acquirePacket(nullptr, false)->getIndex(); }

int FixedEdge::getPosition(const Node *node) { return 0; }
int FixedEdge::getGraphOutputPosition() { return 0; }

base::EdgeUpdateFlag FixedEdge::update(const Node *node) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (terminate_flag_) {
    return base::kEdgeUpdateFlagTerminate;
  } else {
    return base::kEdgeUpdateFlagComplete;
  }
}

bool FixedEdge::requestTerminate() {
  std::lock_guard<std::mutex> lock(mutex_);
  terminate_flag_ = true;
  return true;
}

}  // namespace dag
}  // namespace nndeploy