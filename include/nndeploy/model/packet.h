
#ifndef _NNDEPLOY_MODEL_PACKET_H_
#define _NNDEPLOY_MODEL_PACKET_H_

#include "nndeploy/base/common.h"
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
#include "nndeploy/model/preprocess/params.h"

namespace nndeploy {
namespace model {

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

  device::Buffer *getBuffer(int index = 0) {
    if (index >= buffers_.size()) {
      return nullptr;
    }
    return buffers_[index];
  }
  device::Mat *getMat(int index = 0) {
    if (index >= mats_.size()) {
      return nullptr;
    }
    return mats_[index];
  }
  device::Tensor *getTensor(int index = 0) {
    if (index >= tensors_.size()) {
      return nullptr;
    }
    return tensors_[index];
  }
  base::Param *getParam(int index = 0) {
    if (index >= params_.size()) {
      return nullptr;
    }
    return params_[index];
  }
  void *getAnything(int index) {
    if (index >= anythings_.size()) {
      return nullptr;
    }
    return anythings_[index];
  }
#ifdef ENABLE_NNDEPLOY_OPENCV
  cv::Mat *getCvMat(int index = 0) {
    if (index >= cv_mats_.size()) {
      return nullptr;
    }
    return cv_mats_[index];
  }
#endif

  std::vector<device::Buffer *> getAllBuffer() { return buffers_; }
  std::vector<device::Mat *> getAllMat() { return mats_; }
  std::vector<device::Tensor *> getAllTensor() { return tensors_; }
  std::vector<base::Param *> getAllParam() { return params_; }
  std::vector<void *> getAllAnything() { return anythings_; }
#ifdef ENABLE_NNDEPLOY_OPENCV
  std::vector<cv::Mat *> getAllCvMat() { return cv_mats_; }
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

  void clear() {
    clearBuffer();
    clearMat();
    clearTensor();
    clearParam();
    clearAnything();
#ifdef ENABLE_NNDEPLOY_OPENCV
    clearCvMat();
#endif
  }
  void clearBuffer() { buffers_.clear(); }
  void clearMat() { mats_.clear(); }
  void clearTensor() { tensors_.clear(); }
  void clearParam() { params_.clear(); }
  void clearAnything() { anythings_.clear(); }
#ifdef ENABLE_NNDEPLOY_OPENCV
  void clearCvMat() { cv_mats_.clear(); }
#endif

  size_t sizeBuffer() { return buffers_.size(); }
  size_t sizeMat() { return mats_.size(); }
  size_t sizeTensor() { return tensors_.size(); }
  size_t sizeParam() { return params_.size(); }
  size_t sizeAnything() { return anythings_.size(); }
#ifdef ENABLE_NNDEPLOY_OPENCV
  size_t sizeCvMat() { return cv_mats_.size(); }
#endif

  bool empty() {
    bool flag = emptyBuffer() && emptyMat() && emptyTensor() && emptyParam() &&
                emptyAnything();
#ifdef ENABLE_NNDEPLOY_OPENCV
    flag = flag && emptyCvMat();
#endif
    return flag;
  }
  bool emptyBuffer() { return buffers_.empty(); }
  bool emptyMat() { return mats_.empty(); }
  bool emptyTensor() { return tensors_.empty(); }
  bool emptyParam() { return params_.empty(); }
  bool emptyAnything() { return anythings_.empty(); }
#ifdef ENABLE_NNDEPLOY_OPENCV
  bool emptyCvMat() { return cv_mats_.empty(); }
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

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_PACKET_H_ */
