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

base::Status PipelineEdge::construct() {
  consumers_size_ = consumers_.size();
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

base::Status PipelineEdge::set(device::Buffer *buffer, int index,
                               bool is_external) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(buffer, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  return status;
}
base::Status PipelineEdge::set(device::Buffer &buffer, int index) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(buffer, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
device::Buffer *PipelineEdge::create(device::Device *device,
                                     const device::BufferDesc &desc,
                                     int index) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  device::Buffer *ret_value = dp->create(device, desc, index);
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
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

  return dp->getBuffer();
}
device::Buffer *PipelineEdge::getGraphOutputBuffer() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getDataPacket(nullptr);
  }
  if (dp == nullptr) {
    return nullptr;
  }
  return dp->getBuffer();
}

base::Status PipelineEdge::set(device::Mat *mat, int index, bool is_external) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(device::Mat &mat, int index) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(mat, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
device::Mat *PipelineEdge::create(device::Device *device,
                                  const device::MatDesc &desc, int index,
                                  const std::string &name) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  device::Mat *ret_value = dp->create(device, desc, index, name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n");

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Mat *mat) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if ((*iter)->notifyWritten(mat)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This mat[%p] is error.\n", mat);
  }
  return is_notify;
}
device::Mat *PipelineEdge::getMat(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");
  return dp->getMat();
}
device::Mat *PipelineEdge::getGraphOutputMat() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getDataPacket(nullptr);
  }
  if (dp == nullptr) {
    return nullptr;
  }
  return dp->getMat();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineEdge::set(cv::Mat *cv_mat, int index, bool is_external) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(cv_mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(cv::Mat &cv_mat, int index) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(cv_mat, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
cv::Mat *PipelineEdge::getCvMat(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

  return dp->getCvMat();
}
cv::Mat *PipelineEdge::getGraphOutputCvMat() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getDataPacket(nullptr);
  }
  if (dp == nullptr) {
    return nullptr;
  }
  return dp->getCvMat();
}
#endif

base::Status PipelineEdge::set(device::Tensor *tensor, int index,
                               bool is_external) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(tensor, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(device::Tensor &tensor, int index) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(tensor, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
device::Tensor *PipelineEdge::create(device::Device *device,
                                     const device::TensorDesc &desc, int index,
                                     const std::string &name) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  device::Tensor *ret_value = dp->create(device, desc, index, name);
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
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

  return dp->getTensor();
}
device::Tensor *PipelineEdge::getGraphOutputTensor() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getDataPacket(nullptr);
  }
  if (dp == nullptr) {
    return nullptr;
  }
  return dp->getTensor();
}

base::Status PipelineEdge::set(base::Param *param, int index,
                               bool is_external) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(param, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(base::Param &param, int index) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(param, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Param *PipelineEdge::getParam(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

  return dp->getParam();
}
base::Param *PipelineEdge::getGraphOutputParam() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getDataPacket(nullptr);
  }
  if (dp == nullptr) {
    return nullptr;
  }
  return dp->getParam();
}

base::Status PipelineEdge::set(void *anything, int index, bool is_external) {
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");

  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(anything, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
void *PipelineEdge::getAnything(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

  return dp->getAnything();
}
void *PipelineEdge::getGraphOutputAnything() {
  PipelineDataPacket *dp = nullptr;
  base::EdgeUpdateFlag update_flag = update(nullptr);
  if (update_flag == base::kEdgeUpdateFlagTerminate) {
    NNDEPLOY_LOGI("User voluntarily terminates.\n");
  } else if (update_flag == base::kEdgeUpdateFlagError) {
    NNDEPLOY_LOGI("getGraphOutput update error.\n");
  } else {
    dp = getDataPacket(nullptr);
  }
  if (dp == nullptr) {
    return nullptr;
  }
  return dp->getAnything();
}

int PipelineEdge::getIndex(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getDataPacket error.\n");
    return -1;
  }
  int index = dp->getIndex();
  return index;
}
int PipelineEdge::getGraphOutputIndex() {
  PipelineDataPacket *dp = getDataPacket(nullptr);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getDataPacket error.\n");
    return -1;
  }
  int index = dp->getIndex();
  return index;
}

int PipelineEdge::getPosition(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getDataPacket error.\n");
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

  // update
  for (auto &iter : to_consume_index_) {
    iter.second -= count;
  }
  to_consume_index_[tmp_node]++;  // 会导致下一次数据没有的时候线程一直在等待
  iter = data_packets_.begin();
  for (int i = 0; i < count; i++) {
    delete (*iter);
    iter++;
  }
  data_packets_.erase(data_packets_.begin(), iter);  // 销毁不会被使用到的数据

  consuming_dp_[tmp_node] = dp;
  return base::kEdgeUpdateFlagComplete;
}

PipelineDataPacket *PipelineEdge::getDataPacket(const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  auto iter = consuming_dp_.find(tmp_node);
  if (iter == consuming_dp_.end()) {
    NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
    return nullptr;
  }
  if (iter->second == nullptr) {
    NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
    NNDEPLOY_LOGE("dp is nullptr!\n");
    return nullptr;
  }
  return iter->second;
}

}  // namespace dag
}  // namespace nndeploy