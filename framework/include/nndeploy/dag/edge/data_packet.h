#ifndef _NNDEPLOY_DAG_EDGE_DATA_PACKET_H_
#define _NNDEPLOY_DAG_EDGE_DATA_PACKET_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/dag/base.h"

namespace nndeploy {
namespace dag {

/**
 * @brief
 * 数据包的状态
 * # 未创建
 * # 已创建
 * # 已写入
 * # 没被消费
 * # 正在被消费
 * # 被消费过
 * # 被所有节点消费
 */
class DataPacket : public base::NonCopyable {
 public:
  DataPacket();
  virtual ~DataPacket();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  device::Buffer *create(device::Device *device, const device::BufferDesc &desc,
                         int index);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  cv::Mat *create(int rows, int cols, int type, const cv::Vec3b& value,
                  int index);
  virtual bool notifyWritten(cv::Mat *cv_mat);
  virtual cv::Mat *getCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         int index, const std::string &name);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor();

  virtual base::Status set(base::Param *param, int index, bool is_external);
  template <typename T, typename... Args, typename std::enable_if<std::is_base_of<base::Param, T>::value, int>::type = 0>
  base::Param *create(int index, Args &&...args){
    base::Status status = base::kStatusCodeOk;
    T *param = nullptr;
    if (anything_ == nullptr) {
      param = new T(std::forward<Args>(args)...);
    } else {
      destory();
      param = new T(std::forward<Args>(args)...);
    }
    if (param == nullptr) {
      NNDEPLOY_LOGE("Failed to create param.\n");
      return nullptr;
    }
    is_external_ = false;
    index_ = index;
    flag_ = EdgeTypeFlag::kParam;
    written_ = false;
    anything_ = (void *)(param);
    type_info_ = &typeid(T);
    deleter_ = [](void* d) { delete static_cast<T*>(d); };
    return param;
  }
  virtual bool notifyWritten(base::Param *param);
  virtual base::Param *getParam();

  template <typename T>
  base::Status setAny(T *t, int index, bool is_external = true){
    base::Status status = base::kStatusCodeOk;
    if (anything_ == nullptr) {
      anything_ = (void *)(t);
    } else {
      destory();
      anything_ = (void *)(t);
    }
    is_external_ = is_external;
    index_ = index;
    flag_ = EdgeTypeFlag::kAny;
    written_ = false;
    deleter_ = [](void* d) { delete static_cast<T*>(d); };
    type_info_ = &typeid(T);
    return status;
  }
  template <typename T, typename... Args>
  T *createAny(int index, Args &&...args){
    base::Status status = base::kStatusCodeOk;
    T *t = nullptr;
    if (anything_ == nullptr) {
      t = new T(std::forward<Args>(args)...);
    } else {
      destory();
      t = new T(std::forward<Args>(args)...);
    }
    if (t == nullptr) {
      NNDEPLOY_LOGE("Failed to create param.\n");
      return nullptr;
    }
    is_external_ = false;
    index_ = index;
    flag_ = EdgeTypeFlag::kAny;
    written_ = false;
    anything_ = (void *)(t);
    type_info_ = &typeid(T);
    deleter_ = [](void* d) { delete static_cast<T*>(d); };
    return t;
  }
  bool notifyAnyWritten(void *anything){
    if (anything == anything_) {
      written_ = true;
      return true;
    } else {
      return false;
    }
  }
  template <typename T>
  T *getAny(){
    if (flag_ != EdgeTypeFlag::kAny) {
      return nullptr;
    }
    if (typeid(T) != *type_info_) {
      return nullptr;
    }
    return static_cast<T*>(anything_);
  }

  base::Status takeDataPacket(DataPacket *packet);

  int getIndex();

 protected:
  void destory();

 protected:
  bool is_external_ = true;
  int index_ = -1;
  EdgeTypeFlag flag_ = EdgeTypeFlag::kNone;
  bool written_ = false;
  void *anything_ = nullptr;
  std::function<void(void*)> deleter_;
  std::type_info* type_info_;
};

class PipelineDataPacket : public DataPacket {
 public:
  PipelineDataPacket(int consumers_size);
  virtual ~PipelineDataPacket();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual bool notifyWritten(cv::Mat *cv_mat);
  virtual cv::Mat *getCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor();

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual bool notifyWritten(base::Param *param);
  virtual base::Param *getParam();

  template <typename T>
  base::Status setAny(T *t, int index, bool is_external = true){
    std::unique_lock<std::mutex> lock(mutex_);
    base::Status status = DataPacket::setAny(t, index, is_external);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "DataPacket::setAny failed!\n");
    cv_.notify_all();
    return status;
  }
  bool notifyAnyWritten(void *anything){
    std::unique_lock<std::mutex> lock(mutex_);
    bool status = DataPacket::notifyAnyWritten(anything);
    if (status) {
      cv_.notify_all();
    }
    return status;
  }
  template <typename T>
  T *getAny(){
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return written_; });
    return DataPacket::getAny<T>();
  }

  void increaseConsumersSize();
  void increaseConsumersCount();

  int getConsumersSize();
  int getConsumersCount();

 protected:
  std::mutex mutex_;
  std::condition_variable cv_;
  int consumers_size_ = 0;
  int consumers_count_ = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif
