#include "nndeploy/dag/edge/pipeline_edge.h"

#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

TypeEdgeRegister<TypeEdgeCreator<PipelineEdge>> g_pipeline_edge_register(
    kEdgeTypePipeline);

PipelineEdge::PipelineEdge(ParallelType paralle_type)
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
  // if (consumers_.empty()) {
  //   to_consume_index_.insert({nullptr, 0});
  //   consuming_dp_.insert({nullptr, nullptr});
  // }
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
  bool flag = false;
  flag = markGraphOutput();
  if (!flag) {
    return nullptr;
  }
  flag = update(nullptr);
  if (!flag) {
    if (terminate_flag_) {
      NNDEPLOY_LOGI("User voluntarily terminates.\n");
    }
    return nullptr;
  }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

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
  bool flag = false;
  flag = markGraphOutput();
  if (!flag) {
    return nullptr;
  }
  flag = update(nullptr);
  if (!flag) {
    if (terminate_flag_) {
      NNDEPLOY_LOGI("User voluntarily terminates.\n");
    }
    return nullptr;
  }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

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
  bool flag = false;
  flag = markGraphOutput();
  if (!flag) {
    return nullptr;
  }
  flag = update(nullptr);
  if (!flag) {
    if (terminate_flag_) {
      NNDEPLOY_LOGI("User voluntarily terminates.\n");
    }
    return nullptr;
  }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

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
  bool flag = false;
  flag = markGraphOutput();
  if (!flag) {
    return nullptr;
  }
  flag = update(nullptr);
  if (!flag) {
    if (terminate_flag_) {
      NNDEPLOY_LOGI("User voluntarily terminates.\n");
    }
    return nullptr;
  }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

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
  bool flag = false;
  flag = markGraphOutput();
  if (!flag) {
    return nullptr;
  }
  flag = update(nullptr);
  if (!flag) {
    if (terminate_flag_) {
      NNDEPLOY_LOGI("User voluntarily terminates.\n");
    }
    return nullptr;
  }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

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
  bool flag = false;
  flag = markGraphOutput();
  if (!flag) {
    return nullptr;
  }
  flag = update(nullptr);
  if (!flag) {
    if (terminate_flag_) {
      NNDEPLOY_LOGI("User voluntarily terminates.\n");
    }
    return nullptr;
  }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(
      dp, "PipelineDataPacket getDataPacket error.\n");

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
  // bool flag = false;
  // flag = markGraphOutput();
  // if (!flag) {
  //   return -1;
  // }
  PipelineDataPacket *dp = getDataPacket(nullptr);
  if (dp == nullptr) {
    NNDEPLOY_LOGE("PipelineDataPacket getDataPacket error.\n");
    return -1;
  }
  int index = dp->getIndex();
  return index;
}

bool PipelineEdge::requestTerminate() {
  std::unique_lock<std::mutex> lock(mutex_);
  terminate_flag_ = true;
  cv_.notify_all();
  return true;
}

bool PipelineEdge::checkNode(const Node *node) {
  if (node == nullptr) {
    return true;
  } else if (std::find(consumers_.begin(), consumers_.end(), node) !=
             consumers_.end()) {
    return true;
  } else {
    if (node != nullptr) {
      Node *tmp_node = const_cast<Node *>(node);
      NNDEPLOY_LOGE("This node[%s] is error.\n", tmp_node->getName().c_str());
    } else {
      NNDEPLOY_LOGE("This node is error.\n");
    }
    return false;
  }
}

bool PipelineEdge::markGraphOutput() {
  std::unique_lock<std::mutex> lock(mutex_);
  std::call_once(once_, [this]() {
    this->consumers_size_ = this->consumers_size_ + 1;
    this->to_consume_index_.insert({nullptr, 0});
    this->consuming_dp_.insert({nullptr, nullptr});
    for (auto iter : this->data_packets_) {
      (iter)->increaseConsumersSize();
    }
  });
  return true;
}

/**
 * @brief Get the Consumer Node Edge Data Packet object
 *
 * @param node
 * @return PipelineDataPacket*
 * @note 用于获取消费者节点的数据包，对应节点的输入边
 */
bool PipelineEdge::update(const Node *node) {
  Node *tmp_node = const_cast<Node *>(node);
  if (!checkNode(tmp_node)) {
    NNDEPLOY_LOGE("This node is error.\n");
    return false;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, tmp_node] {
    return to_consume_index_[tmp_node] < data_packets_.size() ||
           terminate_flag_;  // 消费者需求的数据已存在，否则等待最新数据  ||
                             // 数据被消耗结束
  });
  if (terminate_flag_) {
    return false;
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
  return true;
}
/**
 * @brief Get the Graph Output Edge Data Packet object
 *
 * @param node
 * @return PipelineDataPacket*
 * @note 用于获取图的输出节点的数据包
 * @perf 这里还可以再优化性能
 */
// bool PipelineEdge::updateGraphOutputData() {
//   std::unique_lock<std::mutex> lock(mutex_);
//   cv_.wait(lock, [this] {
//     bool flag = false;
//     if (data_packets_.empty()) {
//       flag = false;
//     }
//     for (auto iter : data_packets_) {
//       if (iter->getConsumersCount() == consumers_size_) {
//         flag = true;
//         break;
//       }
//     }
//     return flag || terminate_flag_;
//   });

//   if (terminate_flag_) {
//     return false;
//   }

//   // find
//   int count = 0;
//   PipelineDataPacket *dp = nullptr;
//   auto iter = data_packets_.begin();
//   for (; iter != data_packets_.end(); ++iter) {
//     if ((*iter)->getConsumersCount() == consumers_size_) {
//       dp = (*iter);
//       (*iter)->increaseConsumersCount();
//       break;
//     }
//     count++;
//   }

//   // update
//   iter = data_packets_.begin();
//   for (int i = 0; i < count; i++) {
//     delete (*iter);
//     iter++;
//   }
//   data_packets_.erase(data_packets_.begin(), iter);

//   // 返回值
//   consuming_dp_[nullptr] = dp;
//   return true;
// }

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