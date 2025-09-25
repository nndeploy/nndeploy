
#ifndef _NNDEPLOY_DAG_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/base.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief The names of Edge, Mat, and Tensor need to be consistent.
 * @goal
 * 1. Similar to std::any functionality, can store data of any type
 * 2. Supports memory management, creates specific data structures,
 * automatically releases those structures
 * 3. How to better support Python
 * 4. 队列最大值
 * 5. 移除index
 * 6. 在节点内部，可随时获取当前的输入数据，并且不更新队列位置
 * 7. 在节点内部，可随时获取当前的输出数据，并且不更新队列位置
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  Edge();
  Edge(const std::string &name);
  virtual ~Edge();

  std::string getName();

  /**
   * @brief 设置队列最大值，控制边缘缓冲区的最大容量
   *
   * @param max_size 队列的最大容量，必须大于0，默认为16
   * @return base::Status
   * 成功返回Status::kSuccess，参数无效返回Status::kInvalidParam
   * @note 必须在construct()调用前设置，否则将返回Status::kUninitialized
   */
  base::Status setQueueMaxSize(int queue_max_size);
  /**
   * @brief 获取队列最大值
   *
   * @return int 当前队列的最大容量，默认值为16
   */
  int getQueueMaxSize();

  /**
   * @brief Set the Parallel Type object
   *
   * @param paralle_type
   * @return base::Status
   * @note 在construct之前，调用该函数，内部创建出具体的edge
   */
  base::Status setParallelType(const base::ParallelType &paralle_type);
  base::ParallelType getParallelType();

  base::Status construct();

  base::Status set(device::Buffer *buffer, bool is_external = true);
  base::Status set(device::Buffer &buffer);
  device::Buffer *create(device::Device *device,
                         const device::BufferDesc &desc);
  bool notifyWritten(device::Buffer *buffer);
  device::Buffer *getBuffer(const Node *node);
  device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, bool is_external = true);
  base::Status set(cv::Mat &cv_mat);
  cv::Mat *create(int rows, int cols, int type, const cv::Vec3b &value);
  bool notifyWritten(cv::Mat *cv_mat);
  cv::Mat *getCvMat(const Node *node);
  cv::Mat *getGraphOutputCvMat();
#endif

  base::Status set(device::Tensor *tensor, bool is_external = true);
  base::Status set(device::Tensor &tensor);
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         std::string tensor_name = "");
  bool notifyWritten(device::Tensor *tensor);
  device::Tensor *getTensor(const Node *node);
  device::Tensor *getGraphOutputTensor();

  base::Status set(base::Param *param, bool is_external = true);
  base::Status set(base::Param &param);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<base::Param, T>::value,
                                    int>::type = 0>
  base::Param *create(Args &&...args) {
    this->setTypeInfo<T>();
    return abstact_edge_->create<T>(std::forward<Args>(args)...);
  }
  bool notifyWritten(base::Param *param);
  base::Param *getParam(const Node *node);
  base::Param *getGraphOutputParam();

  template <typename T>
  base::Status set(T *t, bool is_external = true) {
    this->setTypeInfo<T>();
    return abstact_edge_->set<T>(t, is_external);
  }
  template <typename T>
  base::Status set(T &t) {
    this->setTypeInfo<T>();
    return this->set(&t, true);
  }
  template <typename T, typename... Args>
  T *create(Args &&...args) {
    this->setTypeInfo<T>();
    return abstact_edge_->create<T>(std::forward<Args>(args)...);
  }
  template <typename T>
  bool notifyWritten(T *t) {
    return abstact_edge_->notifyWritten<T>(t);
  }
  template <typename T>
  T *get(const Node *node) {
    if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
      std::unique_lock<std::mutex> lock(type_info_mutex_);
      type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
    }
    if (!type_info_->isType<T>()) {
      // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
      return nullptr;
    }
    return abstact_edge_->get<T>(node);
  }
  template <typename T>
  T *getGraphOutput() {
    if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
      std::unique_lock<std::mutex> lock(type_info_mutex_);
      type_info_cv_.wait(lock, [this]() { return type_info_ != nullptr; });
    }
    if (!type_info_->isType<T>()) {
      // NNDEPLOY_LOGE("typeid(T) is not *type_info_");
      return nullptr;
    }
    return abstact_edge_->getGraphOutput<T>();
  }

  template <typename PY_WRAPPER, typename T>
  base::Status set4py(PY_WRAPPER *wrapper, T *t, bool is_external = true) {
    this->setTypeInfo<T>();
    return abstact_edge_->set4py<PY_WRAPPER, T>(wrapper, t, is_external);
  }

  int64_t getIndex(const Node *node);
  int64_t getGraphOutputIndex();
  void resetIndex();

  int getPosition(const Node *node);
  int getGraphOutputPosition();

  base::EdgeUpdateFlag update(const Node *node);

  /**
   * @brief
   *
   * @return true
   * @return false
   * @note must be called after the graph is initialized
   */
  bool markGraphOutput();

  base::Status increaseProducers(std::vector<Node *> &producers);
  base::Status increaseConsumers(std::vector<Node *> &consumers);
  std::vector<Node *> getProducers();
  std::vector<Node *> getConsumers();

  bool requestTerminate();

  template <typename T>
  base::Status setTypeInfo() {
    if (type_info_ == nullptr) {
      type_info_ = std::make_shared<EdgeTypeInfo>();
      type_info_->setType<T>();
      // NNDEPLOY_LOGI("setTypeInfo<%s>\n",
      // type_info_->getTypeName().c_str());
    } else {
      type_info_->setType<T>();
      // NNDEPLOY_LOGI("setTypeInfo<%s>\n",
      // type_info_->getTypeName().c_str());
    }
    if (getParallelType() == base::ParallelType::kParallelTypePipeline) {
      type_info_cv_.notify_all();
    }
    return base::kStatusCodeOk;
  }
  base::Status setTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info);
  std::shared_ptr<EdgeTypeInfo> getTypeInfo();
  void setTypeName(const std::string &type_name);
  std::string getTypeName();

  template <typename T>
  bool checkTypeInfo() {
    EdgeTypeInfo other_type_info;
    other_type_info.setType<T>();
    return *type_info_ == other_type_info;
  }
  bool checkTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info);

 private:
  std::string name_;
  AbstractEdge *abstact_edge_ = nullptr;
  std::mutex type_info_mutex_;
  std::condition_variable type_info_cv_;
  std::shared_ptr<EdgeTypeInfo> type_info_;
  int queue_max_size_ = 16;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
