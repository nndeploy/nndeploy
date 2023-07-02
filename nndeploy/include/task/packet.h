#ifndef _NNDEPLOY_SOURCE_TASK_PACKET_H_
#define _NNDEPLOY_SOURCE_TASK_PACKET_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/opencv_include.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/pre_process/params.h"

namespace nndeploy {
namespace task {

class NNDEPLOY_CC_API Packet {
 public:
  Packet(const std::string &name = "") : name_(name) {}
  virtual ~Packet() {}

  std::string getName() { return name_; }

  void set(device::Buffer *buffer, int index = 0) {
    if (index >= buffers_.size()) {
      buffers_.resize(index + 1);
    }
    buffers_[index] = buffer;
  }
  void set(device::Buffer &buffer, int index = 0) {
    if (index >= buffers_.size()) {
      buffers_.resize(index + 1);
    }
    buffers_[index] = &buffer;
  }
  void set(device::Mat *mat, int index = 0) {
    if (index >= mats_.size()) {
      mats_.resize(index + 1);
    }
    mats_[index] = mat;
  }
  void set(device::Mat &mat, int index = 0) {
    if (index >= mats_.size()) {
      mats_.resize(index + 1);
    }
    mats_[index] = &mat;
  }
  void set(device::Tensor *tensor, int index = 0) {
    if (index >= tensors_.size()) {
      tensors_.resize(index + 1);
    }
    tensors_[index] = tensor;
  }
  void set(device::Tensor &tensor, int index = 0) {
    if (index >= tensors_.size()) {
      tensors_.resize(index + 1);
    }
    tensors_[index] = &tensor;
  }
  void set(base::Param *param, int index = 0) {
    if (index >= params_.size()) {
      params_.resize(index + 1);
    }
    params_[index] = param;
  }
  void set(base::Param &param, int index = 0) {
    if (index >= params_.size()) {
      params_.resize(index + 1);
    }
    params_[index] = &param;
  }
  void set(void *anything, int index = 0) {
    if (index >= anythings_.size()) {
      anythings_.resize(index + 1);
    }
    anythings_[index] = anything;
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  void set(cv::Mat *cv_mat, int index = 0) {
    if (index >= cv_mats_.size()) {
      cv_mats_.resize(index + 1);
    }
    cv_mats_[index] = cv_mat;
  }
  void set(cv::Mat &cv_mat, int index = 0) {
    if (index >= cv_mats_.size()) {
      cv_mats_.resize(index + 1);
    }
    cv_mats_[index] = &cv_mat;
  }
#endif

  template <typename T>
  T *get(int index = 0)();
  template <Device::Buffer>
  T *get(int index = 0)() {
    if (index >= buffers_.size()) {
      return nullptr;
    }
    return buffers_[index];
  }
  template <Device::Mat>
  T *get(int index = 0)() {
    if (index >= mats_.size()) {
      return nullptr;
    }
    return mats_[index];
  }
  template <Device::Tensor>
  T *get(int index = 0)() {
    if (index >= tensors_.size()) {
      return nullptr;
    }
    return tensors_[index];
  }
  template <base::Param>
  T *get(int index = 0)() {
    if (index >= params_.size()) {
      return nullptr;
    }
    return params_[index];
  }
  template <void>
  T *get(int index = 0)() {
    if (index >= anythings_.size()) {
      return nullptr;
    }
    return anythings_[index];
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  template <cv::Mat>
  T *get(int index = 0)() {
    if (index >= cv_mats_.size()) {
      return nullptr;
    }
    return cv_mats_[index];
  }
#endif

  template <typename T>
  std::vector<T *> getAll() {}
  template <Device::Buffer>
  std::vector<T *> getAll()() {
    return buffers_;
  }
  template <Device::Mat>
  std::vector<T *> getAll()() {
    return mats_;
  }
  template <Device::Tensor>
  std::vector<T *> getAll()() {
    return tensors_;
  }
  template <base::Param>
  std::vector<T *> getAll()() {
    return params_;
  }
  template <void>
  std::vector<T *> getAll()() {
    return anythings_;
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  template <cv::Mat>
  std::vector<T *> getAll()() {
    return cv_mats_;
  }
#endif

  void push_back(device::Buffer *buffer) { buffers_.push_back(buffer); }
  void push_back(device::Buffer &buffer) { buffers_.push_back(&buffer); }
  void push_back(device::Mat *mat) { mats_.push_back(mat); }
  void push_back(device::Mat &mat) { mats_.push_back(&mat); }
  void push_back(device::Tensor *tensor) { tensors_.push_back(tensor); }
  void push_back(device::Tensor &tensor) { tensors_.push_back(&tensor); }
  void push_back(base::Param *param) { params_.push_back(param); }
  void push_back(base::Param &param) { params_.push_back(&param); }
  void push_back(void *anything) { anythings_.push_back(anything); }
#ifdef NNDEPLOY_ENABLE_OPENCV
  void push_back(cv::Mat *cv_mat) { cv_mats_.push_back(cv_mat); }
  void push_back(cv::Mat &cv_mat) { cv_mats_.push_back(&cv_mat); }
#endif

  void erase(device::Buffer *buffer) {
    auto it = std::find(buffers_.begin(), buffers_.end(), buffer);
    if (it != buffers_.end()) {
      buffers_.erase(it);
    }
  }
  void erase(device::Buffer &buffer) {
    auto it = std::find(buffers_.begin(), buffers_.end(), &buffer);
    if (it != buffers_.end()) {
      buffers_.erase(it);
    }
  }
  void erase(device::Mat *mat) {
    auto it = std::find(mats_.begin(), mats_.end(), mat);
    if (it != mats_.end()) {
      mats_.erase(it);
    }
  }
  void erase(device::Mat &mat) {
    auto it = std::find(mats_.begin(), mats_.end(), &mat);
    if (it != mats_.end()) {
      mats_.erase(it);
    }
  }
  void erase(device::Tensor *tensor) {
    auto it = std::find(tensors_.begin(), tensors_.end(), tensor);
    if (it != tensors_.end()) {
      tensors_.erase(it);
    }
  }
  void erase(device::Tensor &tensor) {
    auto it = std::find(tensors_.begin(), tensors_.end(), &tensor);
    if (it != tensors_.end()) {
      tensors_.erase(it);
    }
  }
  void erase(base::Param *param) {
    auto it = std::find(params_.begin(), params_.end(), param);
    if (it != params_.end()) {
      params_.erase(it);
    }
  }
  void erase(base::Param &param) {
    auto it = std::find(params_.begin(), params_.end(), &param);
    if (it != params_.end()) {
      params_.erase(it);
    }
  }
  void erase(void *anything) {
    auto it = std::find(anythings_.begin(), anythings_.end(), anything);
    if (it != anythings_.end()) {
      anythings_.erase(it);
    }
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  void erase(cv::Mat *cv_mat) {
    auto it = std::find(cv_mats_.begin(), cv_mats_.end(), cv_mat);
    if (it != cv_mats_.end()) {
      cv_mats_.erase(it);
    }
  }
  void erase(cv::Mat &cv_mat) {
    auto it = std::find(cv_mats_.begin(), cv_mats_.end(), &cv_mat);
    if (it != cv_mats_.end()) {
      cv_mats_.erase(it);
    }
  }
#endif

  template <typename T>
  void clear();
  template <Task::Packet>
  void clear() {
    clear<Device::Buffer>();
    clear<Device::Mat>();
    clear<Device::Tensor>();
    clear<base::Param>();
    clear<void>();
#ifdef NNDEPLOY_ENABLE_OPENCV
    clear<cv::Mat>();
#endif
  }
  template <Device::Buffer>
  void clear() {
    buffers_.clear();
  }
  template <Device::Mat>
  void clear() {
    mats_.clear();
  }
  template <Device::Tensor>
  void clear() {
    tensors_.clear();
  }
  template <base::Param>
  void clear() {
    params_.clear();
  }
  template <void>
  void clear() {
    anythings_.clear();
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  template <cv::Mat>
  void clear() {
    cv_mats_.clear();
  }
#endif

  template <typename T>
  bool empty();
  template <Task::Packet>
  bool empty() {
    bool flag = empty<Device::Buffer>() && empty<Device::Mat>() &&
                empty<Device::Tensor>() && empty<base::Param>() &&
                empty<void>();
#ifdef NNDEPLOY_ENABLE_OPENCV
    flag = flag && empty<cv::Mat>();
#endif
    return flag;
  }
  template <Device::Buffer>
  bool empty() {
    return buffers_.empty();
  }
  template <Device::Mat>
  bool empty() {
    return mats_.empty();
  }
  template <Device::Tensor>
  bool empty() {
    return tensors_.empty();
  }
  template <base::Param>
  bool empty() {
    return params_.empty();
  }
  template <void>
  bool empty() {
    return anythings_.empty();
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  template <cv::Mat>
  bool empty() {
    return cv_mats_.empty();
  }
#endif

  template <typename T>
  size_t size()();
  template <Device::Buffer>
  size_t size()() {
    return buffers_.size();
  }
  template <Device::Mat>
  size_t size()() {
    return mats_.size();
  }
  template <Device::Tensor>
  size_t size()() {
    return tensors_.size();
  }
  template <base::Param>
  size_t size()() {
    return params_.size();
  }
  template <void>
  size_t size()() {
    return anythings_.size();
  }
#ifdef NNDEPLOY_ENABLE_OPENCV
  template <cv::Mat>
  size_t size()() {
    return cv_mats_.size();
  }
#endif

 private:
  std::string name_;

  std::vector<device::Buffer *> buffers_;
  std::vector<device::Mat *> mats_;
  std::vector<device::Tensor *> tensors_;
  std::vector<base::Param *> params_;
  std::vector<void *> anythings_;
#ifdef NNDEPLOY_ENABLE_OPENCV
  std::vector<cv::Mat *> cv_mats_;
#endif
};

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_TASK_PACKET_H_ */
