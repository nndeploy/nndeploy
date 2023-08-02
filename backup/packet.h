#ifndef D55826A2_8B51_416B_AEAB_CBC5E867D254
#define D55826A2_8B51_416B_AEAB_CBC5E867D254
#ifndef ECFC206B_4D40_471F_B71D_0EC48AEDAFD6
#define ECFC206B_4D40_471F_B71D_0EC48AEDAFD6
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
  T *Packet::get(int index = 0);

  template <typename T>
  std::vector<T *> Packet::getAll() {}

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

  template <typename T>
  bool empty();

  template <typename T>
  size_t size();

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

template <>
device::Buffer *Packet::get<device::Buffer>(int index) {
  if (index >= buffers_.size()) {
    return nullptr;
  }
  return buffers_[index];
}
template <>
device::Mat *Packet::get<device::Mat>(int index) {
  if (index >= mats_.size()) {
    return nullptr;
  }
  return mats_[index];
}
template <>
device::Tensor *Packet::get<device::Tensor>(int index) {
  if (index >= tensors_.size()) {
    return nullptr;
  }
  return tensors_[index];
}
template <>
base::Param *Packet::get<base::Param>(int index) {
  if (index >= params_.size()) {
    return nullptr;
  }
  return params_[index];
}
template <>
void *Packet::get<void>(int index) {
  if (index >= anythings_.size()) {
    return nullptr;
  }
  return anythings_[index];
}
#ifdef ENABLE_NNDEPLOY_OPENCV
template <>
cv::Mat *Packet::get<cv::Mat>(int index) {
  if (index >= cv_mats_.size()) {
    return nullptr;
  }
  return cv_mats_[index];
}
#endif

template <>
std::vector<device::Buffer *> Packet::getAll<device::Buffer>() {
  return buffers_;
}
template <>
std::vector<device::Mat *> Packet::getAll<device::Mat>() {
  return mats_;
}
template <>
std::vector<device::Tensor *> Packet::getAll<device::Tensor>() {
  return tensors_;
}
template <>
std::vector<base::Param *> Packet::getAll<base::Param>() {
  return params_;
}
template <>
std::vector<void *> Packet::getAll<void>() {
  return anythings_;
}
#ifdef ENABLE_NNDEPLOY_OPENCV
template <>
std::vector<cv::Mat *> Packet::getAll<cv::Mat>() {
  return cv_mats_;
}
#endif

template <>
void Packet::clear<Packet>() {
  Packet::clear<device::Buffer>();
  Packet::clear<device::Mat>();
  Packet::clear<device::Tensor>();
  Packet::clear<base::Param>();
  Packet::clear<void>();
#ifdef ENABLE_NNDEPLOY_OPENCV
  Packet::clear<cv::Mat>();
#endif
}
template <>
void Packet::clear<device::Buffer>() {
  buffers_.clear();
}
template <>
void Packet::clear<device::Mat>() {
  mats_.clear();
}
template <>
void Packet::clear<device::Tensor>() {
  tensors_.clear();
}
template <>
void Packet::clear<base::Param>() {
  params_.clear();
}
template <>
void Packet::clear<void>() {
  anythings_.clear();
}
#ifdef ENABLE_NNDEPLOY_OPENCV
template <>
void Packet::clear<cv::Mat>() {
  cv_mats_.clear();
}
#endif

template <>
size_t Packet::size<device::Buffer>() {
  return buffers_.size();
}
template <>
size_t Packet::size<device::Mat>() {
  return mats_.size();
}
template <>
size_t Packet::size<device::Tensor>() {
  return tensors_.size();
}
template <>
size_t Packet::size<base::Param>() {
  return params_.size();
}
template <>
size_t Packet::size<void>() {
  return anythings_.size();
}
#ifdef ENABLE_NNDEPLOY_OPENCV
template <>
size_t Packet::size<cv::Mat>() {
  return cv_mats_.size();
}
#endif

template <>
bool Packet::empty<Packet>() {
  bool flag = Packet::empty<device::Buffer>() && Packet::empty<device::Mat>() &&
              Packet::empty<device::Tensor>() && Packet::empty<base::Param>() &&
              Packet::empty<void>();
#ifdef ENABLE_NNDEPLOY_OPENCV
  flag = flag && Packet::empty<cv::Mat>();
#endif
  return flag;
}
template <>
bool Packet::empty<device::Buffer>() {
  return buffers_.empty();
}
template <>
bool Packet::empty<device::Mat>() {
  return mats_.empty();
}
template <>
bool Packet::empty<device::Tensor>() {
  return tensors_.empty();
}
template <>
bool Packet::empty<base::Param>() {
  return params_.empty();
}
template <>
bool Packet::empty<void>() {
  return anythings_.empty();
}
#ifdef ENABLE_NNDEPLOY_OPENCV
template <>
bool Packet::empty<cv::Mat>() {
  return cv_mats_.empty();
}
#endif

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_TASK_PACKET_H_ */

#endif /* ECFC206B_4D40_471F_B71D_0EC48AEDAFD6 */

#endif /* D55826A2_8B51_416B_AEAB_CBC5E867D254 */
