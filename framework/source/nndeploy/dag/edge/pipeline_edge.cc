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
  if (queueSizeUnlocked() == 0) {
    return true;
  }
  PipelineDataPacket *latest = backUnlocked();
  return latest == nullptr || latest->empty();
}

PipelineEdge::~PipelineEdge() {
  consumers_size_ = 0;

  while (queueSizeUnlocked() > 0) {
    PipelineDataPacket *dp = popFrontUnlocked();
    delete dp;
  }
  data_queue_.clear();

  consuming_dp_.clear();
  to_consume_index_.clear();
}

base::Status PipelineEdge::setQueueMaxSize(int queue_max_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_max_size <= 0) {
    queue_max_size = 1;
  }
  queue_max_size_ = queue_max_size;
  size_t required_capacity = std::max(queueLimit(), queueSizeUnlocked());
  data_queue_.reserve(required_capacity);
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::setQueueOverflowPolicy(
    base::QueueOverflowPolicy policy, int drop_count) {
  std::lock_guard<std::mutex> lock(mutex_);
  overflow_policy_ = policy;
  drop_count_ = drop_count <= 0 ? 1 : drop_count;
  return base::kStatusCodeOk;
}

base::Status PipelineEdge::construct() {
  consumers_size_ = static_cast<int>(consumers_.size());
  for (auto iter : consumers_) {
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
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  // set
  base::Status status = dp->set(buffer, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  // 上锁
  {
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);

    this->increaseIndex();
    dp->setIndex(index_);

    pushBackUnlocked(dp);
  }

  cv_.notify_all();

  return status;
}
device::Buffer *PipelineEdge::create(device::Device *device,
                                     const device::BufferDesc &desc) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");

  {
    // 上锁
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    this->increaseIndex();
    dp->setIndex(index_);
    pushBackUnlocked(dp);
  }

  // create
  device::Buffer *ret_value = dp->create(device, desc);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");

  cv_.notify_all();

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Buffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  size_t queue_size = queueSizeUnlocked();
  for (size_t i = 0; i < queue_size; ++i) {
    PipelineDataPacket *dp = atUnlocked(queue_size - 1 - i);
    if (dp != nullptr && dp->notifyWritten(buffer)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This buffer[%p] is error.\n", buffer);
  }
  return is_notify;
}
device::Buffer *PipelineEdge::getBuffer(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getPipelineDataPacket error.\n");

  Node *tmp_node = const_cast<Node *>(node);
  if (consuming_dp_.find(tmp_node) != consuming_dp_.end()) {
    return dp->getBuffer();
  } else {
    return dp->getBufferDirect();
  }
}
device::Buffer *PipelineEdge::getGraphOutputBuffer() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getPipelineDataPacket(nullptr);
  }
  if (dp == nullptr) {
    NNDEPLOY_LOGE(
        "PipelineDataPacket is null, this edge is not output edge.\n");
    return nullptr;
  }
  return dp->getBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineEdge::set(cv::Mat *cv_mat, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  // set
  base::Status status = dp->set(cv_mat, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  {
    // 上锁
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    this->increaseIndex();
    dp->setIndex(index_);

    pushBackUnlocked(dp);
  }

  cv_.notify_all();

  return status;
}
cv::Mat *PipelineEdge::create(int rows, int cols, int type,
                              const cv::Vec3b &value) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");

  {
    // 上锁
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    this->increaseIndex();
    dp->setIndex(index_);
    pushBackUnlocked(dp);
  }

  // create
  cv::Mat *ret_value = dp->create(rows, cols, type, value);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");
  cv_.notify_all();

  return ret_value;
}
bool PipelineEdge::notifyWritten(cv::Mat *cv_mat) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  size_t queue_size = queueSizeUnlocked();
  for (size_t i = 0; i < queue_size; ++i) {
    PipelineDataPacket *dp = atUnlocked(queue_size - 1 - i);
    if (dp != nullptr && dp->notifyWritten(cv_mat)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This cv_mat[%p] is error.\n", cv_mat);
  }
  return is_notify;
}
cv::Mat *PipelineEdge::getCvMat(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getPipelineDataPacket error.\n");

  Node *tmp_node = const_cast<Node *>(node);
  if (consuming_dp_.find(tmp_node) != consuming_dp_.end()) {
    return dp->getCvMat();
  } else {
    return dp->getCvMatDirect();
  }
}
cv::Mat *PipelineEdge::getGraphOutputCvMat() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getPipelineDataPacket(nullptr);
  }
  if (dp == nullptr) {
    NNDEPLOY_LOGE(
        "PipelineDataPacket is null, this edge is not output edge.\n");
    return nullptr;
  }
  return dp->getCvMat();
}
#endif

base::Status PipelineEdge::set(device::Tensor *tensor, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  // set
  base::Status status = dp->set(tensor, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  {
    // 上锁
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    this->increaseIndex();
    dp->setIndex(index_);
    pushBackUnlocked(dp);
  }

  cv_.notify_all();

  return base::kStatusCodeOk;
}
device::Tensor *PipelineEdge::create(device::Device *device,
                                     const device::TensorDesc &desc,
                                     const std::string &name) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");

  {
    // 上锁
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    this->increaseIndex();
    dp->setIndex(index_);
    pushBackUnlocked(dp);
  }

  // create
  device::Tensor *ret_value = dp->create(device, desc, name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");
  cv_.notify_all();

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Tensor *tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  size_t queue_size = queueSizeUnlocked();
  for (size_t i = 0; i < queue_size; ++i) {
    PipelineDataPacket *dp = atUnlocked(queue_size - 1 - i);
    if (dp != nullptr && dp->notifyWritten(tensor)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This tensor[%p] is error.\n", tensor);
  }
  return is_notify;
}
device::Tensor *PipelineEdge::getTensor(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getPipelineDataPacket error.\n");

  Node *tmp_node = const_cast<Node *>(node);
  if (consuming_dp_.find(tmp_node) != consuming_dp_.end()) {
    return dp->getTensor();
  } else {
    return dp->getTensorDirect();
  }
}
device::Tensor *PipelineEdge::getGraphOutputTensor() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getPipelineDataPacket(nullptr);
  }
  if (dp == nullptr) {
    NNDEPLOY_LOGE(
        "PipelineDataPacket is null, this edge is not output edge.\n");
    return nullptr;
  }
  return dp->getTensor();
}

base::Status PipelineEdge::takeDataPacket(DataPacket *data_packet) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  // take
  base::Status status = dp->takeDataPacket(data_packet);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket take error.\n");
  // take
  // base::Status status = dp->takeDataPacket(data_packet);
  // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
  //                        "PipelineDataPacket take error.\n");

  // 上锁
  {
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    pushBackUnlocked(dp);
  }

  cv_.notify_all();

  return status;
}
bool PipelineEdge::notifyWritten(void *anything) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  size_t queue_size = queueSizeUnlocked();
  for (size_t i = 0; i < queue_size; ++i) {
    PipelineDataPacket *dp = atUnlocked(queue_size - 1 - i);
    if (dp != nullptr && dp->notifyWritten(anything)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This anything[%p] is error.\n", anything);
  }
  return is_notify;
}
DataPacket *PipelineEdge::getDataPacket(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getPipelineDataPacket error.\n");

  return static_cast<DataPacket *>(dp);
}
DataPacket *PipelineEdge::getGraphOutputDataPacket() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getPipelineDataPacket(nullptr);
  }
  if (dp == nullptr) {
    NNDEPLOY_LOGE(
        "PipelineDataPacket is null, this edge is not output edge.\n");
    return nullptr;
  }
  return static_cast<DataPacket *>(dp);
}

base::Status PipelineEdge::set(base::Param *param, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  // set
  base::Status status = dp->set(param, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  {
    // 上锁
    std::unique_lock<std::mutex> lock(mutex_);
    waitForSpaceLocked(lock);
    this->increaseIndex();
    dp->setIndex(index_);
    pushBackUnlocked(dp);
  }
  cv_.notify_all();

  return status;
}
bool PipelineEdge::notifyWritten(base::Param *param) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  size_t queue_size = queueSizeUnlocked();
  for (size_t i = 0; i < queue_size; ++i) {
    PipelineDataPacket *dp = atUnlocked(queue_size - 1 - i);
    if (dp != nullptr && dp->notifyWritten(param)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This param[%p] is error.\n", param);
  }
  return is_notify;
}
base::Param *PipelineEdge::getParam(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getPipelineDataPacket error.\n");

  Node *tmp_node = const_cast<Node *>(node);
  if (consuming_dp_.find(tmp_node) != consuming_dp_.end()) {
    return dp->getParam();
  } else {
    return dp->getParamDirect();
  }
}
base::Param *PipelineEdge::getGraphOutputParam() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getPipelineDataPacket(nullptr);
  }
  if (dp == nullptr) {
    NNDEPLOY_LOGE(
        "PipelineDataPacket is null, this edge is not output edge.\n");
    return nullptr;
  }
  return dp->getParam();
}

int64_t PipelineEdge::getIndex(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getPipelineDataPacket error.\n");
    return -1;
  }
  int64_t index = dp->getIndex();
  return index;
}
int64_t PipelineEdge::getGraphOutputIndex() {
  PipelineDataPacket *dp = getPipelineDataPacket(nullptr);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getPipelineDataPacket error.\n");
    return -1;
  }
  int64_t index = dp->getIndex();
  return index;
}

int PipelineEdge::getPosition(const Node *node) {
  PipelineDataPacket *dp = getPipelineDataPacket(node);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getPipelineDataPacket error.\n");
    return -1;
  }
  int position = 0;
  size_t queue_size = queueSizeUnlocked();
  for (size_t i = 0; i < queue_size; ++i) {
    if (dp == atUnlocked(i)) {
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
base::EdgeUpdateFlag PipelineEdge::update(const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  if (!checkNode(tmp_node)) {
    NNDEPLOY_LOGE("This node is error.\n");
    return base::kEdgeUpdateFlagError;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, tmp_node] {
    auto iter = to_consume_index_.find(tmp_node);
    int index_value = iter != to_consume_index_.end() ? iter->second : 0;
    size_t index = index_value < 0 ? 0 : static_cast<size_t>(index_value);
    return index < queueSizeUnlocked() ||
           terminate_flag_;  // 消费者需求的数据已存在，否则等待最新数据
                             // || 数据被消耗结束
  });
  if (terminate_flag_) {
    return base::kEdgeUpdateFlagTerminate;
  }

  // find
  PipelineDataPacket *dp = nullptr;
  int index = to_consume_index_[tmp_node];
  if (index < 0) {
    index = 0;
  }
  int count = 0;
  for (int i = 0; i < index; i++) {
    PipelineDataPacket *candidate = atUnlocked(static_cast<size_t>(i));
    if (candidate != nullptr &&
        candidate->getConsumersCount() == consumers_size_) {
      count++;
    }
  }
  dp = atUnlocked(static_cast<size_t>(index));
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket is null!\n");
    return base::kEdgeUpdateFlagError;
  }
  dp->increaseConsumersCount();
  consuming_dp_[tmp_node] = dp;

  // update
  size_t real_count = 0;
  for (int i = 0; i < count; i++) {
    PipelineDataPacket *candidate = frontUnlocked();
    if (candidate == nullptr) {
      break;
    }
    bool delete_flag = true;
    for (auto consuming_dp : consuming_dp_) {
      if (consuming_dp.second == candidate) {
        delete_flag = false;
        break;
      }
    }
    if (delete_flag) {
      popFrontUnlocked();
      real_count++;
    } else {
      break;
    }
  }

  if (real_count > 0) {
    for (auto &iter : to_consume_index_) {
      iter.second -= static_cast<int>(real_count);
    }
    size_t new_size = queueSizeUnlocked();
    size_t prev_size = new_size + real_count;
    size_t limit = queueLimit();
    if (prev_size >= limit && new_size < limit) {
      queue_cv_.notify_all();
    }
  }
  // 让下一次数据没有的时候线程一直在等待
  to_consume_index_[tmp_node]++;

  return base::kEdgeUpdateFlagComplete;
}

PipelineDataPacket *PipelineEdge::getPipelineDataPacket(const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  if (consuming_dp_.find(tmp_node) != consuming_dp_.end()) {
    auto iter = consuming_dp_.find(tmp_node);
    if (iter->second == nullptr) {
      NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
      NNDEPLOY_LOGE("dp is nullptr!\n");
      return nullptr;
    }
    return iter->second;
  } else if (std::find(producers_.begin(), producers_.end(), tmp_node) !=
             producers_.end()) {
    return backUnlocked();
  } else {
    NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
    return nullptr;
  }
}

void PipelineEdge::pushBackUnlocked(PipelineDataPacket *dp) {
  size_t required_capacity = std::max(queueLimit(), queueSizeUnlocked() + 1);
  data_queue_.reserve(required_capacity);
  data_queue_.pushBack(dp);
}

PipelineDataPacket *PipelineEdge::atUnlocked(size_t index) const {
  return data_queue_.at(index);
}

PipelineDataPacket *PipelineEdge::frontUnlocked() const {
  return data_queue_.front();
}

PipelineDataPacket *PipelineEdge::backUnlocked() const {
  return data_queue_.back();
}

PipelineDataPacket *PipelineEdge::popFrontUnlocked() {
  return data_queue_.popFront();
}

size_t PipelineEdge::queueSizeUnlocked() const { return data_queue_.size(); }

size_t PipelineEdge::queueLimit() const {
  return static_cast<size_t>(queue_max_size_ <= 0 ? 1 : queue_max_size_);
}

void PipelineEdge::waitForSpaceLocked(std::unique_lock<std::mutex> &lock) {
  size_t limit = queueLimit();
  if (limit == 0) {
    return;
  }

  switch (overflow_policy_) {
    case base::QueueOverflowPolicy::kQueueOverflowPolicyNodeBackpressure:
      if (!hasGraphOutputConsumer()) {
        queue_cv_.wait(lock,
                       [this]() { return queueSizeUnlocked() < queueLimit(); });
      }
      break;
    case base::QueueOverflowPolicy::kQueueOverflowPolicyAllBackpressure:
      queue_cv_.wait(lock,
                     [this]() { return queueSizeUnlocked() < queueLimit(); });
      break;
    case base::QueueOverflowPolicy::kQueueOverflowPolicyDropOldest: {
      size_t normalized_drop = drop_count_ <= 0
                                   ? static_cast<size_t>(1)
                                   : static_cast<size_t>(drop_count_);
      while (queueSizeUnlocked() >= limit) {
        size_t dropped = dropOldestUnlocked(normalized_drop);
        if (dropped == 0) {
          queue_cv_.wait(
              lock, [this]() { return queueSizeUnlocked() < queueLimit(); });
          break;
        }
      }
      break;
    }
    default:
      break;
  }
}

size_t PipelineEdge::dropOldestUnlocked(size_t count) {
  if (count == 0) {
    count = 1;
  }
  size_t dropped = 0;
  while (dropped < count && queueSizeUnlocked() > 0) {
    PipelineDataPacket *candidate = frontUnlocked();
    if (candidate == nullptr) {
      break;
    }
    bool in_use = false;
    for (auto &consuming_dp : consuming_dp_) {
      if (consuming_dp.second == candidate) {
        in_use = true;
        break;
      }
    }
    if (in_use) {
      break;
    }
    PipelineDataPacket *removed = popFrontUnlocked();
    if (removed != nullptr) {
      delete removed;
    }
    ++dropped;
  }

  if (dropped > 0) {
    for (auto &iter : to_consume_index_) {
      iter.second -= static_cast<int>(dropped);
      if (iter.second < 0) {
        iter.second = 0;
      }
    }
    if (queueSizeUnlocked() < queueLimit()) {
      queue_cv_.notify_all();
    }
  }

  return dropped;
}

bool PipelineEdge::hasGraphOutputConsumer() const {
  return std::find(consumers_.begin(), consumers_.end(), nullptr) !=
         consumers_.end();
}

}  // namespace dag
}  // namespace nndeploy
