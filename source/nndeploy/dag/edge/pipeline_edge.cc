#include "nndeploy/dag/edge/pipeline_edge.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineEdge>> g_pipeline_edge_register(
    kEdgeTypePipeline);

PipelineEdge::PipelineEdge(ParallelType paralle_type,
                           std::initializer_list<Node *> producers,
                           std::initializer_list<Node *> consumers)
    : AbstractEdge(paralle_type, producers, consumers) {
  producers_count_ = producers.size();
  consumers_count_ = consumers.size();
  for (auto iter : consumers) {
    consumed_.insert({iter, 0});
  }
}

PipelineEdge::~PipelineEdge() {
  producers_count_ = -1;
  consumers_count_ = -1;

  for (auto iter : data_packets_) {
    delete iter.first;
  }
  data_packets_.clear();

  consumed_.clear();
}

base::Status PipelineEdge::set(device::Buffer *buffer, int index,
                               bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(buffer, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
base::Status PipelineEdge::set(device::Buffer &buffer, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(buffer, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
device::Buffer *PipelineEdge::create(device::Device *device,
                                     const device::BufferDesc &desc,
                                     int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n")
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  device::Buffer *ret_value = dp->create(device, desc, index);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n")

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Buffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  boo is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if (iter.first->notifyWritten(buffer)) {
      is_notify = true;
      break;
    }
  }
  if (!is_notify) {
    NNDEPLOY_LOGE("This buffer[%p] is error.\n", buffer);
  }
  return is_notify;
}
device::Buffer *getBuffer(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getBuffer();
}

base::Status PipelineEdge::set(device::Mat *mat, int index, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
base::Status PipelineEdge::set(device::Mat &mat, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(mat, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
device::Mat *PipelineEdge::create(device::Device *device,
                                  const device::MatDesc &desc, int index,
                                  const std::string &name) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n")
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  device::Mat *ret_value = dp->create(device, desc, index, name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n")

  return ret_value;
}
bool PipelineEdge::notifyWritten(device::Mat *mat) {
  std::lock_guard<std::mutex> lock(mutex_);
  boo is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if (iter.first->notifyWritten(mat)) {
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
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getMat();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineEdge::set(cv::Mat *cv_mat, int index, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(cv_mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
base::Status PipelineEdge::set(cv::Mat &cv_mat, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(cv_mat, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
cv::Mat *PipelineEdge::getCvMat(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getCvMat();
}
#endif

base::Status PipelineEdge::set(device::Tensor *tensor, int index,
                               bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(tensor, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
base::Status PipelineEdge::set(device::Tensor &tensor, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(tensor, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
device::Tensor *PipelineEdge::create(device::Device *device,
                                     const device::TensorDesc &desc,
                                     int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n")
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  device::Tensor *ret_value = dp->create(device, desc, index, name_);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(ret_value,
                                     "PipelineDataPacket create error.\n")
  return ret_value;
}
bool notifyWritten(device::Tensor *tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  boo is_notify = false;
  for (auto iter = data_packets_.rbegin(); iter != data_packets_.rend();
       ++iter) {
    if (iter.first->notifyWritten(tensor)) {
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
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getTensor();
}

base::Status PipelineEdge::set(base::Param *param, int index,
                               bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n");
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(param, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
base::Status PipelineEdge::set(base::Param &param, int index) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n")
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(param, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
base::Param *PipelineEdge::getParam(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getParam();
}

base::Status PipelineEdge::set(void *anything, int index, bool is_external) {
  PipelineDataPacket *dp = new PipelineDataPacket();
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(dp, "PipelineDataPacket is null.\n")
  // 上锁
  std::lock_guard<std::mutex> lock(mutex_);
  data_packets_.push_back({dp, 0});
  cv_.notify_all();
  // set
  base::Status status = dp->set(anything, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "PipelineDataPacket set error.\n")
  return status;
}
void *getAnything(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getAnything();
}

int getIndex(const Node *node) {
  PipelineDataPacket *dp = getDataPacket(node);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n")

  return dp->getIndex();
}

PipelineDataPacket *PipelineEdge::getDataPacket(const Node *node) {
  if (consumers_.empty() && node == nullptr) {
    return getGraphOutputEdgeDataPacket(node);
  } else if (consumers_.find(node) != consumers_.end()) {
    return getConsumerNodeEdgeDataPacket(node);
  } else {
    if (node != nullptr) {
      NNDEPLOY_LOGE("This node[%s] is error.\n", node->getName().c_str());
    } else {
      NNDEPLOY_LOGE("This node is error.\n");
    }
    return nullptr;
  }
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
   *
   * @return std::unique_lock<std::mutex>
   */
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return consumed_[node] < data_packets_.size() });
  // find
  int index = consumed_[node];
  int count = 0;
  auto iter = data_packets_.begin();
  for (int i = 0; i <= index; i++) {
    if (iter.second == consumers_count_) {
      count++;
    }
    iter++;
  }
  iter.second++;
  PipelineDataPacket *dp = iter.first;
  // update
  consumed_[node]++;  // 会导致下一次数据没有的时候线程一直在等待
  for (auto iter : consumed_) {
    iter.second -= count;
  }
  auto iter = data_packets_.begin();
  for (int i = 0; i < count; i++) {
    delete iter.first;
    iter++;
  }
  data_packets_.erase(data_packets_.begin(), iter);

  return dp;
}
/**
 * @brief Get the Graph Output Edge Data Packet object
 *
 * @param node
 * @return PipelineDataPacket*
 * @note 用于获取图的输出节点的数据包
 */
PipelineDataPacket *PipelineEdge::getGraphOutputEdgeDataPacket(
    const Node *node) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] {
    if (data_packets_.empty()) {
      return false;
    }
    for (auto iter : data_packets_) {
      if (iter.second == consumers_) {
        return true;
      }
    }
    return false;
  });
  // find
  int count = 0;
  PipelineDataPacket *dp = nullptr;
  for (iter = data_packets_.begin(); iter != data_packets_.end(); ++iter) {
    if (iter.second == consumers_) {
      PipelineDataPacket *dp = iter.first;
      break;
    }
    count++;
  }
  iter.second++;
  // update
  auto iter = data_packets_.begin();
  for (int i = 0; i < count; i++) {
    delete iter.first;
    iter++;
  }
  data_packets_.erase(data_packets_.begin(), iter);

  return dp;
}

}  // namespace dag
}  // namespace nndeploy