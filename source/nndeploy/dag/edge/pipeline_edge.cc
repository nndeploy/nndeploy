#include "nndeploy/dag/edge/pipeline_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineEdge>> g_pipeline_edge_register(
    kEdgeTypePipeline);

PipelineEdge::PipelineEdge(ParallelType paralle_type,
                           std::vector<Node *> &producers,
                           std::vector<Node *> &consumers)
    : AbstractEdge(paralle_type, producers, consumers) {
  consumers_size_ = consumers.size();
  for (auto iter : consumers) {
    to_consume_index_.insert({iter, 0});
    consuming_dp_.insert({iter, nullptr});
  }
}

PipelineEdge::~PipelineEdge() {
  consumers_size_ = 0;

  for (auto iter : data_packets_) {
    delete iter;
  }
  data_packets_.clear();

  consuming_dp_.clear();
  to_consume_index_.clear();
}

base::Status PipelineEdge::set(device::Buffer *buffer, int index,
                               bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  base::Status status = dp->set(buffer, index, is_external);
  data_packets_.push_back(dp);
  cv_.notify_all();

  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");

  return status;
}
base::Status PipelineEdge::set(device::Buffer &buffer, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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

base::Status PipelineEdge::set(device::Mat *mat, int index, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(device::Mat &mat, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineEdge::set(cv::Mat *cv_mat, int index, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(cv_mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(cv::Mat &cv_mat, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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

  cv::Mat *ret = dp->getCvMat();

  NNDEPLOY_LOGE("cv::Mat = %p!\n", ret);

  return ret;
}
#endif

base::Status PipelineEdge::set(device::Tensor *tensor, int index,
                               bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(tensor, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(device::Tensor &tensor, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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

base::Status PipelineEdge::set(base::Param *param, int index,
                               bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back(dp);
  cv_.notify_all();
  // set
  base::Status status = dp->set(param, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n");
  return status;
}
base::Status PipelineEdge::set(base::Param &param, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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

base::Status PipelineEdge::set(void *anything, int index, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket(consumers_size_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
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

int PipelineEdge::getIndex(const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  auto iter = consuming_dp_.find(tmp_node);
  if (iter == consuming_dp_.end()) {
    NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
    return -1;
  }
  if (iter->second == nullptr) {
    NNDEPLOY_LOGE("node[%s] is error!\n", tmp_node->getName().c_str());
    NNDEPLOY_LOGE("dp is nullptr!\n");
    return -1;
  }
  int index = iter->second->getIndex();
  return index;
}

PipelineDataPacket *PipelineEdge::getDataPacket(const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  PipelineDataPacket *ret_value = nullptr;
  if (consumers_.empty() && node == nullptr) {
    ret_value = getGraphOutputEdgeDataPacket(node);
  } else if (std::find(consumers_.begin(), consumers_.end(), node) !=
             consumers_.end()) {
    ret_value = getConsumerNodeEdgeDataPacket(node);
  } else {
    if (node != nullptr) {
      NNDEPLOY_LOGE("This node[%s] is error.\n", tmp_node->getName().c_str());
    } else {
      NNDEPLOY_LOGE("This node is error.\n");
    }
  }
  consuming_dp_[tmp_node] = ret_value;
  return ret_value;
}

/**
 * @brief Get the Consumer Node Edge Data Packet object
 *
 * @param node
 * @return PipelineDataPacket*
 * @note 用于获取消费者节点的数据包，对应节点的输入边
 */
PipelineDataPacket *PipelineEdge::getConsumerNodeEdgeDataPacket(
    const Node *node) {
  /**
   * @brief 多个线程在调用条件变量的wait方法时会阻塞住
   * notify_one:
   * 此时调用notify_one会随机唤醒一个阻塞的线程，而其余的线程将仍然处于阻塞状态，等待下一次唤醒
   *
   * notify_all:
   * 调用notify_all则会唤醒所有线程，线程会争抢锁，当然只有一个线程会获得到锁，而其余未获得锁的线程也将不再阻塞，而是进入到类似轮询的状态，等待锁资源释放后再去争抢。
   *
   * 假如同时有10个线程阻塞在wait方法上，则需要调用10次notify_one，而仅仅只需要调用1次notify_all
   */
  Node *tmp_node = const_cast<Node *>(node);
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, tmp_node] {
    return to_consume_index_[tmp_node] < data_packets_.size();
  });

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
  to_consume_index_[tmp_node]++;  // 会导致下一次数据没有的时候线程一直在等待
  // for (auto iter : to_consume_index_) {
  //   iter.second -= count;
  // }
  // iter = data_packets_.begin();
  // for (int i = 0; i < count; i++) {
  //   delete (*iter);
  //   iter++;
  // }
  // data_packets_.erase(data_packets_.begin(), iter);

  if (tmp_node != nullptr) {
    NNDEPLOY_LOGE("node name %s!Thread ID: %d.\n", tmp_node->getName().c_str(),
                  std::this_thread::get_id());
    NNDEPLOY_LOGE("data_packets_.size = %d.Thread ID: %d.\n",
                  data_packets_.size(), std::this_thread::get_id());
    auto iter = data_packets_.begin();
    for (int i = 0; i < data_packets_.size(); i++) {
      NNDEPLOY_LOGE("getConsumersCount = %d. size = %d. Thread ID: %d.\n",
                    (*iter)->getConsumersCount(), consumers_size_,
                    std::this_thread::get_id());
      iter++;
    }
  }

  return dp;
}
/**
 * @brief Get the Graph Output Edge Data Packet object
 *
 * @param node
 * @return PipelineDataPacket*
 * @note 用于获取图的输出节点的数据包
 * @perf 这里还可以再优化性能
 */
PipelineDataPacket *PipelineEdge::getGraphOutputEdgeDataPacket(
    const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] {
    if (data_packets_.empty()) {
      return false;
    }
    for (auto iter : data_packets_) {
      if (iter->getConsumersCount() == consumers_size_) {
        return true;
      }
    }
    return false;
  });

  // find
  int count = 0;
  PipelineDataPacket *dp = nullptr;
  auto iter = data_packets_.begin();
  for (; iter != data_packets_.end(); ++iter) {
    if ((*iter)->getConsumersCount() == consumers_size_) {
      dp = (*iter);
      (*iter)->increaseConsumersCount();
      break;
    }
    count++;
  }

  if (tmp_node == nullptr) {
    NNDEPLOY_LOGE("node name!Thread ID: %d.\n", std::this_thread::get_id());
    NNDEPLOY_LOGE("data_packets_.size = %d.Thread ID: %d.\n",
                  data_packets_.size(), std::this_thread::get_id());
    auto iter = data_packets_.begin();
    for (int i = 0; i < data_packets_.size(); i++) {
      NNDEPLOY_LOGE("getConsumersCount = %d. size = %d. Thread ID: %d.\n",
                    (*iter)->getConsumersCount(), consumers_size_,
                    std::this_thread::get_id());
      iter++;
    }
  }

  // update
  // iter = data_packets_.begin();
  // for (int i = 0; i < count; i++) {
  //   delete (*iter);
  //   iter++;
  // }
  // data_packets_.erase(data_packets_.begin(), iter);

  // 返回值
  return dp;
}

}  // namespace dag
}  // namespace nndeploy