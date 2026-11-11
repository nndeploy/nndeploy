
#include "nndeploy/dag/edge/pipeline_edge.h"

#include <algorithm>

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineEdge>> g_pipeline_edge_register(
    base::kEdgeTypePipeline);

PipelineEdge::PipelineEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type) {
  if (queue_max_size_ <= 0) {
    queue_max_size_ = 1;
  }
  data_queue_.reserve(static_cast<size_t>(queue_max_size_));
}

bool PipelineEdge::empty() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto latest = data_queue_.back();
  return !latest || latest->empty();
}

PipelineEdge::~PipelineEdge() = default;

base::Status PipelineEdge::setQueueMaxSize(int queue_max_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_max_size <= 0) {
    queue_max_size = 1;
  }
  queue_max_size_ = queue_max_size;
  size_t required_capacity = std::max(queueLimit(), data_queue_.size());
  data_queue_.reserve(required_capacity);
  queue_cv_.notify_all();
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::setQueueOverflowPolicy(
    base::QueueOverflowPolicy policy, int drop_count) {
  std::lock_guard<std::mutex> lock(mutex_);
  overflow_policy_ = policy;
  drop_count_ = drop_count <= 0 ? 1 : drop_count;
  queue_cv_.notify_all();
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::construct() {
  std::vector<Packet> retired;
  std::lock_guard<std::mutex> lock(mutex_);
  if (terminate_flag_) {
    while (!data_queue_.empty()) retired.push_back(data_queue_.popFront());
    for (auto &entry : consuming_dp_) retired.push_back(std::move(entry.second));
    consuming_dp_.clear(); to_consume_index_.clear(); next_sequence_.clear();
    index_ = -1;
    terminate_flag_ = false;
  }
  consumers_size_ = static_cast<int>(consumers_.size());
  for (auto iter : consumers_) {
    next_sequence_.emplace(iter, 0);
    if (to_consume_index_.find(iter) == to_consume_index_.end()) {
      to_consume_index_.insert({iter, 0});
    }
    if (consuming_dp_.find(iter) == consuming_dp_.end()) {
      consuming_dp_.insert({iter, nullptr});
    }
  }
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::set(device::Buffer *buffer, bool is_external) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  packet->set(buffer, is_external);
  auto status = enqueue(packet);
  if (status != base::kStatusCodeOk) packet->borrowPayload();
  return status;
}
device::Buffer *PipelineEdge::create(device::Device *device,
                                     const device::BufferDesc &desc) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  auto *result = packet->create(device, desc);
  return result && enqueue(packet) == base::kStatusCodeOk ? result : nullptr;
}
bool PipelineEdge::notifyWritten(device::Buffer *buffer) {
  return notifyWritten(static_cast<void *>(buffer));
}
device::Buffer *PipelineEdge::getBuffer(const Node *node) {
  auto packet = acquirePacket(node, false);
  return packet ? packet->DataPacket::getBuffer() : nullptr;
}
device::Buffer *PipelineEdge::getGraphOutputBuffer() {
  auto packet = acquirePacket(nullptr, true, EdgeTypeFlag::kBuffer);
  return packet ? packet->DataPacket::getBuffer() : nullptr;
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineEdge::set(cv::Mat *cv_mat, bool is_external) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  packet->set(cv_mat, is_external);
  auto status = enqueue(packet);
  if (status != base::kStatusCodeOk) packet->borrowPayload();
  return status;
}
cv::Mat *PipelineEdge::create(int rows, int cols, int type,
                              const cv::Vec3b &value) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  auto *result = packet->create(rows, cols, type, value);
  return result && enqueue(packet) == base::kStatusCodeOk ? result : nullptr;
}
bool PipelineEdge::notifyWritten(cv::Mat *cv_mat) {
  return notifyWritten(static_cast<void *>(cv_mat));
}
cv::Mat *PipelineEdge::getCvMat(const Node *node) {
  auto packet = acquirePacket(node, false);
  return packet ? packet->DataPacket::getCvMat() : nullptr;
}
cv::Mat *PipelineEdge::getGraphOutputCvMat() {
  auto packet = acquirePacket(nullptr, true, EdgeTypeFlag::kCvMat);
  return packet ? packet->DataPacket::getCvMat() : nullptr;
}
#endif

base::Status PipelineEdge::set(device::Tensor *tensor, bool is_external) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  packet->set(tensor, is_external);
  auto status = enqueue(packet);
  if (status != base::kStatusCodeOk) packet->borrowPayload();
  return status;
}
device::Tensor *PipelineEdge::create(device::Device *device,
                                     const device::TensorDesc &desc,
                                     const std::string &name) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  auto *result = packet->create(device, desc, name);
  return result && enqueue(packet) == base::kStatusCodeOk ? result : nullptr;
}
bool PipelineEdge::notifyWritten(device::Tensor *tensor) {
  return notifyWritten(static_cast<void *>(tensor));
}
device::Tensor *PipelineEdge::getTensor(const Node *node) {
  auto packet = acquirePacket(node, false);
  return packet ? packet->DataPacket::getTensor() : nullptr;
}
device::Tensor *PipelineEdge::getGraphOutputTensor() {
  auto packet = acquirePacket(nullptr, true, EdgeTypeFlag::kTensor);
  return packet ? packet->DataPacket::getTensor() : nullptr;
}

base::Status PipelineEdge::takeDataPacket(DataPacket *data_packet) {
  if (!data_packet) return base::kStatusCodeErrorInvalidParam;
  return enqueue(std::make_shared<PipelineDataPacket>(consumers_size_), data_packet);
}
bool PipelineEdge::notifyWritten(void *anything) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool notified = false;
  for (size_t i = 0; i < data_queue_.size(); ++i) {
    auto packet = data_queue_.at(i);
    // Reposted aliases refer to the same write, so wake every matching packet.
    notified = packet->notifyWritten(anything) || notified;
  }
  if (notified) {
    // Wake both overflow writers and consumers that skipped this packet and
    // are waiting to retire it before another producer can enter the queue.
    queue_cv_.notify_all();
    cv_.notify_all();
  } else {
    NNDEPLOY_LOGE("This payload[%p] is error.\n", anything);
  }
  return notified;
}
DataPacket *PipelineEdge::getDataPacket(const Node *node) {
  return currentPacket(node).get();
}
DataPacket *PipelineEdge::getGraphOutputDataPacket() {
  return acquirePacket(nullptr, true).get();
}

base::Status PipelineEdge::set(base::Param *param, bool is_external) {
  auto packet = std::make_shared<PipelineDataPacket>(consumers_size_);
  packet->set(param, is_external);
  auto status = enqueue(packet);
  if (status != base::kStatusCodeOk) packet->borrowPayload();
  return status;
}
bool PipelineEdge::notifyWritten(base::Param *param) {
  return notifyWritten(static_cast<void *>(param));
}
base::Param *PipelineEdge::getParam(const Node *node) {
  auto packet = acquirePacket(node, false);
  return packet ? packet->DataPacket::getParam() : nullptr;
}
base::Param *PipelineEdge::getGraphOutputParam() {
  auto packet = acquirePacket(nullptr, true, EdgeTypeFlag::kParam);
  return packet ? packet->DataPacket::getParam() : nullptr;
}

void *PipelineEdge::getGraphOutputPtr() {
  auto packet = acquirePacket(nullptr, true);
  return packet ? packet->getDataPtr() : nullptr;
}

int64_t PipelineEdge::getIndex(const Node *node) {
  auto dp = currentPacket(node);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getPipelineDataPacket error.\n");
    return -1;
  }
  int64_t index = dp->getIndex();
  return index;
}
int64_t PipelineEdge::getGraphOutputIndex() {
  auto dp = currentPacket(nullptr);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getPipelineDataPacket error.\n");
    return -1;
  }
  int64_t index = dp->getIndex();
  return index;
}

int PipelineEdge::getPosition(const Node *node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = consuming_dp_.find(const_cast<Node *>(node));
  auto dp = found == consuming_dp_.end() ? data_queue_.back() : found->second;
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getPipelineDataPacket error.\n");
    return -1;
  }
  int position = 0;
  size_t queue_size = data_queue_.size();
  for (size_t i = 0; i < queue_size; ++i) {
    if (dp == data_queue_.at(i)) {
      break;
    }
    position++;
  }
  return position;
}
int PipelineEdge::getGraphOutputPosition() { return getPosition(nullptr); }

bool PipelineEdge::requestTerminate() {
  std::unique_lock<std::mutex> lock(mutex_);
  terminate_flag_ = true;
  for (size_t i = 0; i < data_queue_.size(); ++i) data_queue_.at(i)->cancel();
  queue_cv_.notify_all();
  cv_.notify_all();
  return true;
}

/**
 * @brief Get the Consumer Node Edge Data Packet object
 *
 * @param node
 * @return PipelineDataPacket*
 * @note 用于获取消费者节点的数据包，对应节点的输入边
 */
base::Status PipelineEdge::skipFrame() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (terminate_flag_) return base::kStatusCodeNodeInterrupt;
  increaseIndex();
  cv_.notify_all();
  return base::kStatusCodeOk;
}

base::EdgeUpdateFlag PipelineEdge::update(const Node *node) {
  Packet packet;
  return nextPacket(node, packet, EdgeTypeFlag::kNone, nullptr);
}

base::EdgeUpdateFlag PipelineEdge::nextPacket(const Node *node, Packet &packet,
                                             EdgeTypeFlag expected, const std::type_info *type) {
  auto *consumer = const_cast<Node *>(node);
  std::vector<Packet> retired;
  std::unique_lock<std::mutex> lock(mutex_);
  if (!checkNode(node)) return base::kEdgeUpdateFlagError;

  // Release the previous slot before waiting for a producer on a bounded queue.
  consuming_dp_[consumer].reset();
  while (true) {
    while (!data_queue_.empty() &&
           data_queue_.front()->getConsumersCount() == consumers_size_) {
      if (dropOldestUnlocked(1, retired) == 0) break;
    }
    queue_cv_.notify_all();
    if (!retired.empty()) {
      lock.unlock();
      retired.clear();  // Payload finalizers can re-enter the edge.
      lock.lock();
    }
    if (terminate_flag_) return base::kEdgeUpdateFlagTerminate;
    auto position = static_cast<size_t>(to_consume_index_[consumer]);
    auto &sequence = next_sequence_[consumer];
    // Gaps represent dropped frames. They carry no payload and consume no
    // queue capacity, but downstream nodes must still settle each frame.
    if (sequence <= index_ &&
        (position >= data_queue_.size() || data_queue_.at(position)->getIndex() > sequence)) {
      ++sequence;
      return base::kEdgeUpdateFlagSkip;
    }
    if (position < data_queue_.size()) break;
    // A write completion can make an already consumed packet reclaimable.
    // Recheck retirement on wakeup, not just whether a new packet arrived.
    cv_.wait(lock);
  }
  auto &index = to_consume_index_[consumer];
  auto next = data_queue_.at(index);
  // A typed probe must not consume a different kind of graph output.
  if (!next->matches(expected, type)) return base::kEdgeUpdateFlagError;
  ++index;
  ++next_sequence_[consumer];
  packet = std::move(next);
  packet->increaseConsumersCount();
  consuming_dp_[consumer] = packet;
  return base::kEdgeUpdateFlagComplete;
}

PipelineEdge::Packet PipelineEdge::currentPacket(const Node *node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto *key = const_cast<Node *>(node);
  auto consumer = consuming_dp_.find(key);
  if (consumer != consuming_dp_.end()) return consumer->second;
  if (std::find(producers_.begin(), producers_.end(), key) != producers_.end())
    return data_queue_.back();
  return nullptr;
}

std::shared_ptr<DataPacket> PipelineEdge::acquirePacket(const Node *node, bool graph_output,
                                                        EdgeTypeFlag expected, const std::type_info *type) {
  Packet packet;
  if (graph_output) {
    base::EdgeUpdateFlag flag;
    do { flag = nextPacket(nullptr, packet, expected, type); }
    while (flag == base::kEdgeUpdateFlagSkip);
    if (flag != base::kEdgeUpdateFlagComplete) return nullptr;
  } else {
    packet = currentPacket(node);
    if (packet && std::find(consumers_.begin(), consumers_.end(), node) == consumers_.end()) return packet;  // Producer filling a new packet.
  }
  return packet && packet->waitUntilWritten() ? packet : nullptr;
}

size_t PipelineEdge::queueLimit() const { return static_cast<size_t>(queue_max_size_); }

base::Status PipelineEdge::enqueue(Packet packet, DataPacket *source) {
  std::vector<Packet> retired;
  // Pin an existing owner before backpressure or dropping can retire it.
  // Declared before the lock so payload finalizers always run outside it.
  Packet previous;
  std::unique_lock<std::mutex> lock(mutex_);
  auto *payload = source ? source->getDataPtr() : packet->getDataPtr();
  auto find_previous = [&] {
    for (size_t i = data_queue_.size(); i > 0; --i) {
      auto candidate = data_queue_.at(i - 1);
      if (candidate->getDataPtr() == payload) return candidate;
    }
    return Packet{};
  };
  previous = find_previous();
  while (!terminate_flag_ && data_queue_.size() >= queueLimit()) {
    if (overflow_policy_ == base::kQueueOverflowPolicyNodeBackpressure &&
        hasGraphOutputConsumer()) break;
    if (overflow_policy_ == base::kQueueOverflowPolicyDropOldest &&
        dropOldestUnlocked(drop_count_, retired)) continue;
    if (!retired.empty()) {
      lock.unlock();
      retired.clear();
      lock.lock();
      continue;
    }
    queue_cv_.wait(lock);
  }
  if (terminate_flag_) return base::kStatusCodeErrorInvalidValue;
  data_queue_.reserve(data_queue_.size() + 1);
  if (source) packet->takeDataPacket(source);  // Ownership transfers on admission.
  // Another producer may have admitted the same payload while we were waiting.
  if (!previous) previous = find_previous();
  packet->retainSamePayload(previous);
  increaseIndex();
  packet->setIndex(index_);
  data_queue_.pushBack(std::move(packet));
  lock.unlock();
  cv_.notify_all();
  return base::kStatusCodeOk;
}

size_t PipelineEdge::dropOldestUnlocked(size_t count, std::vector<Packet> &retired) {
  size_t dropped = 0;
  while (dropped < count && !data_queue_.empty()) {
    auto candidate = data_queue_.front();
    bool in_use = std::any_of(consuming_dp_.begin(), consuming_dp_.end(),
        [&](const auto &entry) { return entry.second == candidate; });
    // create() returns a writable raw pointer. It remains live until the
    // producer publishes it with notifyWritten(), even if nobody has read it.
    if (in_use || !candidate->isWritten()) break;
    retired.push_back(data_queue_.popFront());
    ++dropped;
  }
  if (dropped) {
    for (auto &entry : to_consume_index_)
      entry.second = std::max(0, entry.second - static_cast<int>(dropped));
    queue_cv_.notify_all();
  }
  return dropped;
}

bool PipelineEdge::hasGraphOutputConsumer() const {
  return std::find(consumers_.begin(), consumers_.end(), nullptr) !=
         consumers_.end();
}

}  // namespace dag
}  // namespace nndeploy
