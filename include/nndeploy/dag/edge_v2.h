
#ifndef _NNDEPLOY_DAG_EDGE_V2_H_
#define _NNDEPLOY_DAG_EDGE_V2_H_

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

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API EdgeV2 {
 public:
  enum kEdgeType : int {
    kEdgeTypeBuffer = 0,
    kEdgeTypeMat = 1,
#ifdef ENABLE_NNDEPLOY_OPENCV
    kEdgeTypeCvMat = 2,
#endif
    kEdgeTypeTensor = 4,
    kEdgeTypeParam = 8,

    kEdgeTypeVoid = 1 << 31,
  };
  EdgeV2() {}
  virtual ~EdgeV2() {}

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
  device::Tensor *getTensor(const std::string &name) {
    for (auto tensor : tensors_) {
      if (name == tensor->getName()) {
        return tensor;
      }
    }
    return nullptr;
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

  void push_back(device::Buffer *buffer) { buffers_.emplace_back(buffer); }
  void push_back(device::Buffer &buffer) { buffers_.emplace_back(&buffer); }
  void push_back(device::Mat *mat) { mats_.emplace_back(mat); }
  void push_back(device::Mat &mat) { mats_.emplace_back(&mat); }
  void push_back(device::Tensor *tensor) { tensors_.emplace_back(tensor); }
  void push_back(device::Tensor &tensor) { tensors_.emplace_back(&tensor); }
  void push_back(base::Param *param) { params_.emplace_back(param); }
  void push_back(base::Param &param) { params_.emplace_back(&param); }
  void push_back(void *anything) { anythings_.emplace_back(anything); }
#ifdef ENABLE_NNDEPLOY_OPENCV
  void push_back(cv::Mat *cv_mat) { cv_mats_.emplace_back(cv_mat); }
  void push_back(cv::Mat &cv_mat) { cv_mats_.emplace_back(&cv_mat); }
#endif

  void clear() { anything_.clear(); }

  size_t batchSize() { return batch_size_; }
  size_t total() { return anything_.size(); }
  size_t size() {
    size_t size = std::floor(anything_.size() / batch_size_);
    return size;
  }

  bool empty() {
    bool flag = anything_.empty();
    return flag;
  }

 private:
  std::string name_;
  bool is_external_;
  kEdgeType type_;
  bool is_pipeline_parallel_ = false;
  // 比如多batch推理
  size_t batch_size_ = 1;
  std::vector<void *> anything_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_EDGE_V2_H_ */
