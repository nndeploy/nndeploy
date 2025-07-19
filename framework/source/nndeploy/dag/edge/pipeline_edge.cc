#include "nndeploy/dag/edge/pipeline_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineEdge>> g_pipeline_edge_register(
    base::kEdgeTypePipeline);

PipelineEdge::PipelineEdge(base::ParallelType paralle_type)
    : AbstractEdge(paralle_type) {}

PipelineEdge::~PipelineEdge() {
  consumers_size_ = 0;

  for (auto iter : data_packets_) {
    delete iter;
  }
  data_packets_.clear();

  consuming_dp_.clear();
  to_consume_index_.clear();
}

base::Status PipelineEdge::setQueueMaxSize(int queue_max_size) {
  queue_max_size_ = queue_max_size;
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
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);

  data_packets_.push_back(dp);
  cv_.notify_all();

  // set
  base::Status status = dp->set(buffer, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  return status;
}
device::Buffer *PipelineEdge::create(device::Device *device,
                                     const device::BufferDesc &desc) {
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);
  data_packets_.push_back(dp);
  cv_.notify_all();

  // create
  device::Buffer *ret_value = dp->create(device, desc);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Buffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if ((*iter)->notifyWritten(buffer)) {
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
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);

  data_packets_.push_back(dp);
  cv_.notify_all();

  // set
  base::Status status = dp->set(cv_mat, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  return status;
}
cv::Mat *PipelineEdge::create(int rows, int cols, int type,
                              const cv::Vec3b &value) {
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);
  data_packets_.push_back(dp);
  cv_.notify_all();

  // create
  cv::Mat *ret_value = dp->create(rows, cols, type, value);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");

  return ret_value;
}
bool PipelineEdge::notifyWritten(cv::Mat *cv_mat) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if ((*iter)->notifyWritten(cv_mat)) {
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
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);

  data_packets_.push_back(dp);
  cv_.notify_all();

  // set
  base::Status status = dp->set(tensor, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  return base::kStatusCodeOk;
}
device::Tensor *PipelineEdge::create(device::Device *device,
                                     const device::TensorDesc &desc,
                                     const std::string &name) {
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);
  data_packets_.push_back(dp);
  cv_.notify_all();

  // create
  device::Tensor *ret_value = dp->create(device, desc, name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Tensor *tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if ((*iter)->notifyWritten(tensor)) {
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
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();

  // take
  base::Status status = dp->takeDataPacket(data_packet);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket take error.\n");

  return status;
}
bool PipelineEdge::notifyWritten(void *anything) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if ((*iter)->notifyWritten(anything)) {
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
  // 上锁
  std::unique_lock<std::mutex> lock(mutex_);
  if (std::find(consumers_.begin(), consumers_.end(), nullptr) ==
      consumers_.end()) {
    queue_cv_.wait(lock,
                   [this]() { return data_packets_.size() < queue_max_size_; });
  }

  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  this->increaseIndex();
  dp->setIndex(index_);

  data_packets_.push_back(dp);
  cv_.notify_all();

  // set
  base::Status status = dp->set(param, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  return status;
}
bool PipelineEdge::notifyWritten(base::Param *param) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if ((*iter)->notifyWritten(param)) {
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
  for (auto iter : data_packets_) {
    if (dp == iter) {
      break;
    } else {
      position++;
    }
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
    return to_consume_index_[tmp_node] < data_packets_.size() ||
           terminate_flag_;  // 消费者需求的数据已存在，否则等待最新数据  ||
                             // 数据被消耗结束
  });
  if (terminate_flag_) {
    return base::kEdgeUpdateFlagTerminate;
  }

  // find
  PipelineDataPacket *dp = nullptr;
  int index = to_consume_index_[tmp_node];
  int count = 0;
  auto iter = data_packets_.begin();
  for (int i = 0; i < index; i++) {
    if ((*iter)->getConsumersCount() == consumers_size_) {
      count++;
    }
    iter++;
  }
  dp = (*iter);
  dp->increaseConsumersCount();
  consuming_dp_[tmp_node] = dp;

  // update
  int real_count = 0;
  iter = data_packets_.begin();
  for (int i = 0; i < count; i++) {
    bool delete_flag = true;
    for (auto consuming_dp : consuming_dp_) {
      if (consuming_dp.second == *iter) {
        delete_flag = false;
        break;
      }
    }
    if (delete_flag) {
      delete (*iter);
      iter++;
      real_count++;
    } else {
      break;
    }
  }

  // 销毁不会被使用到的数据
  if (real_count > 0) {
    // if (tmp_node != nullptr && tmp_node->getName() == "encode_node_") {
    //   NNDEPLOY_LOGE("encode_node_ real_count[%d]: data_packets_.size[%d].\n",
    //                 real_count, data_packets_.size());
    // }
    data_packets_.erase(data_packets_.begin(), iter);
    for (auto &iter : to_consume_index_) {
      iter.second -= real_count;
    }
    if (data_packets_.size() + real_count >= queue_max_size_ &&
        data_packets_.size() < queue_max_size_) {
      queue_cv_.notify_all();
    }
    // queue_cv_.notify_all();
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
    return data_packets_.back();
  } else {
    NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
    return nullptr;
  }
}

}  // namespace dag
}  // namespace nndeploy