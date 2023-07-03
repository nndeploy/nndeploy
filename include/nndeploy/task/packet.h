#ifndef _NNDEPLOY_TASK_PACKET_H_
#define _NNDEPLOY_TASK_PACKET_H_

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/task/pre_process/params.h"

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
#ifdef ENABLE_NNDEPLOY_OPENCV
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
  T *get(int index = 0);
  template <>
  device::Buffer *get<device::Buffer>(int index) {
    if (index >= buffers_.size()) {
      return nullptr;
    }
    return buffers_[index];
  }
  template <>
  device::Mat *get<device::Mat>(int index) {
    if (index >= mats_.size()) {
      return nullptr;
    }
    return mats_[index];
  }
  template <>
  device::Tensor *get<device::Tensor>(int index) {
    if (index >= tensors_.size()) {
      return nullptr;
    }
    return tensors_[index];
  }
  template <>
  base::Param *get<base::Param>(int index) {
    if (index >= params_.size()) {
      return nullptr;
    }
    return params_[index];
  }
  template <>
  void *get<void>(int index) {
    if (index >= anythings_.size()) {
      return nullptr;
    }
    return anythings_[index];
  }
#ifdef ENABLE_NNDEPLOY_OPENCV
  template <>
  cv::Mat *get<cv::Mat>(int index) {
    if (index >= cv_mats_.size()) {
      return nullptr;
    }
    return cv_mats_[index];
  }
#endif

  template <typename T>
  std::vector<T *> getAll() {}
  template <>
  std::vector<device::Buffer *> getAll<device::Buffer>() {
    return buffers_;
  }
  template <>
  std::vector<device::Mat *> getAll<device::Mat>() {
    return mats_;
  }
  template <>
  std::vector<device::Tensor *> getAll<device::Tensor>() {
    return tensors_;
  }
  template <>
  std::vector<base::Param *> getAll<base::Param>() {
    return params_;
  }
  template <>
  std::vector<void *> getAll<void>() {
    return anythings_;
  }
#ifdef ENABLE_NNDEPLOY_OPENCV
  template <>
  std::vector<cv::Mat *> getAll<cv::Mat>() {
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
#ifdef ENABLE_NNDEPLOY_OPENCV
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
#ifdef ENABLE_NNDEPLOY_OPENCV
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

  template<>
  void clear<Packet>() {
    clear<device::Buffer>();
    clear<device::Mat>();
    clear<device::Tensor>();
    clear<base::Param>();
    clear<void>();
#ifdef ENABLE_NNDEPLOY_OPENCV
    clear<cv::Mat>();
#endif
  }
  template <>
  void clear<device::Buffer>() {
    buffers_.clear();
  }
  template <>
  void clear<device::Mat>() {
    mats_.clear();
  }
  template <>
  void clear<device::Tensor>() {
    tensors_.clear();
  }
  template <>
  void clear<base::Param>() {
    params_.clear();
  }
  template <>
  void clear<void>() {
    anythings_.clear();
  }
#ifdef ENABLE_NNDEPLOY_OPENCV
  template void clear<cv::Mat>() { cv_mats_.clear(); }
#endif

  template <typename T>
  bool empty();
  template <>
  bool empty<Packet>() {
    bool flag = empty<device::Buffer>() && empty<device::Mat>() &&
                empty<device::Tensor>() && empty<base::Param>() &&
                empty<void>();
#ifdef ENABLE_NNDEPLOY_OPENCV
    flag = flag && empty<cv::Mat>();
#endif
    return flag;
  }
  template <>
  bool empty<device::Buffer>() {
    return buffers_.empty();
  }
  template <>
  bool empty<device::Mat>() {
    return mats_.empty();
  }
  template <>
  bool empty<device::Tensor>() {
    return tensors_.empty();
  }
  template <>
  bool empty<base::Param>() {
    return params_.empty();
  }
  template <>
  bool empty<void>() {
    return anythings_.empty();
  }
#ifdef ENABLE_NNDEPLOY_OPENCV
  template <>
  bool empty<cv::Mat>() {
    return cv_mats_.empty();
  }
#endif

  template <typename T>
  size_t size();
  template <>
  size_t size<device::Buffer>() {
    return buffers_.size();
  }
  template <>
  size_t size<device::Mat>() {
    return mats_.size();
  }
  template <>
  size_t size<device::Tensor>() {
    return tensors_.size();
  }
  template <>
  size_t size<base::Param>() {
    return params_.size();
  }
  template <>
  size_t size<void>() {
    return anythings_.size();
  }
#ifdef ENABLE_NNDEPLOY_OPENCV
  template <>
  size_t size<cv::Mat>() {
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
#ifdef ENABLE_NNDEPLOY_OPENCV
  std::vector<cv::Mat *> cv_mats_;
#endif
};

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_TASK_PACKET_H_ */
