
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
 */
class NNDEPLOY_CC_API Edge : public base::NonCopyable {
 public:
  Edge();
  Edge(const std::string &name);
  virtual ~Edge();

  std::string getName();

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

  base::Status set(device::Buffer *buffer, int index, bool is_external = true);
  base::Status set(device::Buffer &buffer, int index);
  device::Buffer *create(device::Device *device, const device::BufferDesc &desc,
                         int index);
  bool notifyWritten(device::Buffer *buffer);
  device::Buffer *getBuffer(const Node *comsumer);
  device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, int index, bool is_external = true);
  base::Status set(cv::Mat &cv_mat, int index);
  cv::Mat *create(int rows, int cols, int type, const cv::Vec3b &value,
                  int index);
  bool notifyWritten(cv::Mat *cv_mat);
  cv::Mat *getCvMat(const Node *node);
  cv::Mat *getGraphOutputCvMat();
#endif

  base::Status set(device::Tensor *tensor, int index, bool is_external = true);
  base::Status set(device::Tensor &tensor, int index);
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         int index, std::string tensor_name = "");
  bool notifyWritten(device::Tensor *tensor);
  device::Tensor *getTensor(const Node *node);
  device::Tensor *getGraphOutputTensor();

  base::Status set(base::Param *param, int index, bool is_external = true);
  base::Status set(base::Param &param, int index);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<base::Param, T>::value,
                                    int>::type = 0>
  base::Param *create(int index, Args &&...args) {
    this->setTypeInfo<T>();
    return abstact_edge_->create<T>(index, std::forward<Args>(args)...);
  }
  bool notifyWritten(base::Param *param);
  base::Param *getParam(const Node *node);
  base::Param *getGraphOutputParam();

  template <typename T>
  base::Status setAny(T *t, int index, bool is_external = true) {
    this->setTypeInfo<T>();
    return abstact_edge_->setAny<T>(t, index, is_external);
  }
  template <typename T>
  base::Status setAny(T &t, int index) {
    this->setTypeInfo<T>();
    return this->setAny(&t, index, true);
  }
  template <typename T, typename... Args>
  T *createAny(int index, Args &&...args) {
    this->setTypeInfo<T>();
    return abstact_edge_->createAny<T>(index, std::forward<Args>(args)...);
  }
  template <typename T>
  bool notifyAnyWritten(T *t) {
    return abstact_edge_->notifyAnyWritten<T>(t);
  }
  template <typename T>
  T *getAny(const Node *node) {
    return abstact_edge_->getAny<T>(node);
  }
  template <typename T>
  T *getGraphOutputAny() {
    return abstact_edge_->getGraphOutputAny<T>();
  }

  int getIndex(const Node *node);
  int getGraphOutputIndex();

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

  bool requestTerminate();

  template <typename T>
  base::Status setTypeInfo() {
    if (type_info_ == nullptr) {
      type_info_ = std::make_shared<EdgeTypeInfo>();
      type_info_->setType<T>();
    } else {
      type_info_->setType<T>();
    }
    return base::kStatusCodeOk;
  }
  base::Status setTypeInfo(std::shared_ptr<EdgeTypeInfo> type_info);
  std::shared_ptr<EdgeTypeInfo> getTypeInfo();

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
  std::shared_ptr<EdgeTypeInfo> type_info_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
