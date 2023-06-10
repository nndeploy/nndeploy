
#include "nndeploy/source/task/packet.h"

#include "nndeploy/source/task/opencv_include.h"

namespace nndeploy {
namespace task {

Packet::Packet() {}
Packet::Packet(std::string name) : name_(name) {}

Packet::Packet(const std::vector<device::Buffer *> &buffers)
    : buffers_(buffers) {}
Packet::Packet(device::Buffer *buffer) { buffers_.push_back(buffer); }
Packet::Packet(device::Buffer &buffer) { buffers_.push_back(&buffer); }

Packet::Packet(const std::vector<device::Mat *> &mats) : mats_(mats) {}
Packet::Packet(device::Mat *mat) { mats_.push_back(mat); }
Packet::Packet(device::Mat &mat) { mats_.push_back(&mat); }

Packet::Packet(const std::vector<device::Tensor *> &tensors)
    : tensors_(tensors) {}
Packet::Packet(device::Tensor *tensor) { tensors_.push_back(tensor); }
Packet::Packet(device::Tensor &tensor) { tensors_.push_back(&tensor); }

Packet::Packet(const std::vector<base::Param *> &params) : params_(params) {}
Packet::Packet(base::Param *param) { params_.push_back(param); }
Packet::Packet(base::Param &param) { params_.push_back(&param); }

#ifdef NNDEPLOY_ENABLE_OPENCV
Packet::Packet(const std::vector<cv::Mat *> &cv_mats) : cv_mats_(cv_mats) {}
Packet::Packet(cv::Mat *cv_mat) { cv_mats_.push_back(cv_mat); }
Packet::Packet(cv::Mat &cv_mat) { cv_mats_.push_back(&cv_mat); }
#endif

Packet::~Packet() {}

base::Status Packet::setName(std::string name) {
  name_ = name;
  return base::kStatusCodeOk;
}
std::string Packet::getName() { return name_; }

void Packet::add(device::Buffer *buffer) { buffers_.push_back(buffer); }
void Packet::add(const std::vector<device::Buffer *> &buffers) {
  for (auto buffer : buffers) {
    buffers_.push_back(buffer);
  }
}
void Packet::add(device::Buffer &buffer) { buffers_.push_back(&buffer); }

void Packet::add(device::Mat *mat) { mats_.push_back(mat); }
void Packet::add(const std::vector<device::Mat *> &mats) {
  for (auto mat : mats) {
    mats_.push_back(mat);
  }
}
void Packet::add(device::Mat &mat) { mats_.push_back(&mat); }

void Packet::add(device::Tensor *tensor) { tensors_.push_back(tensor); }
void Packet::add(const std::vector<device::Tensor *> &tensors) {
  for (auto tensor : tensors) {
    tensors_.push_back(tensor);
  }
}
void Packet::add(device::Tensor &tensor) { tensors_.push_back(&tensor); }

void Packet::add(base::Param *param) { params_.push_back(param); }
void Packet::add(const std::vector<base::Param *> &params) {
  for (auto param : params) {
    params_.push_back(param);
  }
}
void Packet::add(base::Param &param) { params_.push_back(&param); }

#ifdef NNDEPLOY_ENABLE_OPENCV
void Packet::add(cv::Mat *cv_mat) { cv_mats_.push_back(cv_mat); }
void Packet::add(const std::vector<cv::Mat *> &cv_mats) {
  for (auto cv_mat : cv_mats) {
    cv_mats_.push_back(cv_mat);
  }
}
void Packet::add(cv::Mat &cv_mat) { cv_mats_.push_back(&cv_mat); }
#endif

void Packet::add(void *anything) { anything_ = anything; }

// bool Packet::empty() { return emptyBuffer() && emptyMat() && emptyTensor(); }

bool Packet::emptyBuffer() { return buffers_.size() == 0; }
int Packet::getBufferSize() { return buffers_.size(); }
device::Buffer *Packet::getBuffer() { return buffers_[0]; }
device::Buffer *Packet::getBuffer(int index) { return buffers_[index]; }

bool Packet::emptyMat() { return mats_.size() == 0; }
int Packet::getMatSize() { return mats_.size(); }
device::Mat *Packet::getMat() { return mats_[0]; }
device::Mat *Packet::getMat(int index) { return mats_[index]; }

bool Packet::emptyTensor() { return tensors_.size() == 0; }
int Packet::getTensorSize() { return tensors_.size(); }
device::Tensor *Packet::getTensor() { return tensors_[0]; }
device::Tensor *Packet::getTensor(int index) { return tensors_[index]; }

bool Packet::emptyParam() { return params_.size() == 0; }
int Packet::getParamSize() { return params_.size(); }
base::Param *Packet::getParam() { return params_[0]; }
base::Param *Packet::getParam(int index) { return params_[index]; }

#ifdef NNDEPLOY_ENABLE_OPENCV
bool Packet::emptyCvMat() { return cv_mats_.size() == 0; }
int Packet::getCvMatSize() { return cv_mats_.size(); }
cv::Mat *Packet::getCvMat() { return cv_mats_[0]; }
cv::Mat *Packet::getCvMat(int index) { return cv_mats_[index]; }
#endif

void *Packet::getAnything() { return anything_; }

}  // namespace task
}  // namespace nndeploy
