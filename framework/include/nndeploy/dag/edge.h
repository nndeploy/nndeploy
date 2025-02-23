
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
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

  
/**
 * @brief 输入输出类型信息
 * @note 用于描述输入输出类型信息
 */
class NNDEPLOY_CC_API EdgeTypeInfo {
 public:
  enum class TypeFlag {
    kBuffer = 0,
    kCvMat = 1, 
    kTensor = 2,
    kParam = 3,
    kAny = 4,
  };

  EdgeTypeInfo() : type_(TypeFlag::kBuffer), type_info_("") {}
  ~EdgeTypeInfo() = default;

  // 通用类型
  template <typename T>
  void setType(T* t = nullptr) {
    typedef typename std::decay<T>::type DT;
    if constexpr (std::is_base_of<base::Param, DT>::value) {
      type_ = TypeFlag::kParam;
    } else {
      type_ = TypeFlag::kAny;
    }
    type_info_ = getTypeName<DT>();
    type_ptr_ = &typeid(DT);
    type_holder_ = std::make_shared<TypeHolder<DT>>();
  }

  // Buffer类型特化
  void setType(device::Buffer* t = nullptr) ;

#ifdef ENABLE_NNDEPLOY_OPENCV
  // cv::Mat类型特化
  void setType(cv::Mat* t = nullptr) ;
#endif

  // Tensor类型特化
  void setType(device::Tensor* t = nullptr) ;

  TypeFlag getType() const { return type_; }

  std::string getTypeInfo() const { return type_info_; }

  const std::type_info *getTypePtr() const { return type_ptr_; }

  template <typename T>
  bool isType() const {
    return (type_ptr_ != nullptr) && (*type_ptr_ == typeid(T));
  }

  template <typename T, typename... Args>
  T *createType(Args&&... args) {
    if (!isType<T>()) {
      NNDEPLOY_LOGE("Type mismatch in createType\n");
      NNDEPLOY_LOGE(" stored=%s\n", type_ptr_->name());
      NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
      return nullptr;
    }
    return new T(std::forward<Args>(args)...);
  }

  // 获取类型名称的辅助函数
  template <typename T>
  static std::string getTypeName() {
    std::string name = typeid(T).name();
    // 移除编译器相关的前缀和修饰
    size_t pos = name.find_last_of("::");
    if (pos != std::string::npos) {
      name = name.substr(pos + 1);
    }
    return name;
  }

  template <typename T>
  bool checkType() const {
    if (type_ptr_ == nullptr) {
      NNDEPLOY_LOGE("The type info is empty\n");
      NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
      return false;
    }
    if (*type_ptr_ != typeid(T)) {
      NNDEPLOY_LOGE("The stored type mismatch\n");
      NNDEPLOY_LOGE(" stored=%s\n", type_ptr_->name());
      NNDEPLOY_LOGE(" requested=%s\n", typeid(T).name());
      return false;
    }
    return true;
  }

 public:
  // Type holder base class
  struct TypeHolderBase {
    virtual ~TypeHolderBase() = default;
  };

  // Type holder for specific type
  template <typename T>
  struct TypeHolder : TypeHolderBase {
    using Type = T;
  };

  TypeFlag type_;
  std::string type_info_;
  const std::type_info *type_ptr_{nullptr};
  std::shared_ptr<TypeHolderBase> type_holder_;
};

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
  device::Buffer *getBuffer(const Node *node);
  device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, int index, bool is_external = true);
  base::Status set(cv::Mat &cv_mat, int index);
  cv::Mat *create(int rows, int cols, int type, const cv::Vec3b& value,
                  int index);
  bool notifyWritten(cv::Mat *cv_mat);
  cv::Mat *getCvMat(const Node *node);
  cv::Mat *getGraphOutputCvMat();
#endif

  base::Status set(device::Tensor *tensor, int index, bool is_external = true);
  base::Status set(device::Tensor &tensor, int index);
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         int index);
  bool notifyWritten(device::Tensor *tensor);
  device::Tensor *getTensor(const Node *node);
  device::Tensor *getGraphOutputTensor();

  base::Status set(base::Param *param, int index, bool is_external = true);
  base::Status set(base::Param &param, int index);
  template <typename T, typename... Args, typename std::enable_if<std::is_base_of<base::Param, T>::value, int>::type = 0>
  base::Param *create(int index, Args &&...args){
    return abstact_edge_->create<T>(index, std::forward<Args>(args)...);
  }
  bool notifyWritten(base::Param *param);
  base::Param *getParam(const Node *node);
  base::Param *getGraphOutputParam();
  
  template <typename T>
  base::Status setAny(T *t, int index, bool is_external = true){
    return abstact_edge_->setAny<T>(t, index, is_external);
  }
  template <typename T>
  base::Status setAny(T &t, int index, bool is_external = true){
    return abstact_edge_->setAny<T>(&t, index, is_external);
  }
  template <typename T, typename... Args>
  T *createAny(int index, Args &&...args){
    return abstact_edge_->createAny<T>(index, std::forward<Args>(args)...);
  }
  template <typename T>
  bool notifyAnyWritten(T *t){
    return abstact_edge_->notifyAnyWritten<T>(t);
  }
  template <typename T>
  T *getAny(const Node *node){
    return abstact_edge_->getAny<T>(node);
  }
  template <typename T>
  T *getGraphOutputAny(){
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

 private:
  std::string name_;
  AbstractEdge *abstact_edge_ = nullptr;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
