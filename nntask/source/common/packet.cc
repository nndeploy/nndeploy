
#include "nntask/source/common/packet.h"

namespace nntask {
namespace common {

Packet::Packet() {}
Packet::Packet(std::string name) : name_(name) {}

Packet::Packet(const std::vector<nndeploy::device::Buffer *> &buffers)
    : buffers_(buffers) {}
Packet::Packet(nndeploy::device::Buffer *buffer) { buffers_.push_back(buffer); }
Packet::Packet(nndeploy::device::Buffer &buffer) {
  buffers_.push_back(&buffer);
}

Packet::Packet(const std::vector<nndeploy::device::Mat *> &mats)
    : mats_(mats) {}
Packet::Packet(nndeploy::device::Mat *mat) { mats_.push_back(mat); }
Packet::Packet(nndeploy::device::Mat &mat) { mats_.push_back(&mat); }

Packet::Packet(const std::vector<nndeploy::device::Tensor *> &tensors)
    : tensors_(tensors) {}
Packet::Packet(nndeploy::device::Tensor *tensor) { tensors_.push_back(tensor); }
Packet::Packet(nndeploy::device::Tensor &tensor) {
  tensors_.push_back(&tensor);
}

Packet::~Packet() {}

nndeploy::base::Status Packet::setName(std::string name) {
  name_ = name;
  return nndeploy::base::kStatusCodeOk;
}
std::string Packet::getName() { return name_; }

void Packet::add(nndeploy::device::Buffer *buffer) {
  buffers_.push_back(buffer);
}
void Packet::add(const std::vector<nndeploy::device::Buffer *> &buffers) {
  for (auto buffer : buffers) {
    buffers_.push_back(buffer);
  }
}
void Packet::add(nndeploy::device::Buffer &buffer) {
  buffers_.push_back(&buffer);
}

void Packet::add(nndeploy::device::Mat *mat) { mats_.push_back(mat); }
void Packet::add(const std::vector<nndeploy::device::Mat *> &mats) {
  for (auto mat : mats) {
    mats_.push_back(mat);
  }
}
void Packet::add(nndeploy::device::Mat &mat) { mats_.push_back(&mat); }

void Packet::add(nndeploy::device::Tensor *tensor) {
  tensors_.push_back(tensor);
}
void Packet::add(const std::vector<nndeploy::device::Tensor *> &tensors) {
  for (auto tensor : tensors) {
    tensors_.push_back(tensor);
  }
}
void Packet::add(nndeploy::device::Tensor &tensor) {
  tensors_.push_back(&tensor);
}

void Packet::add(nndeploy::base::Param *param) { params_.push_back(param); }
void Packet::add(const std::vector<nndeploy::base::Param *> &params) {
  for (auto param : params) {
    params_.push_back(param);
  }
}
void Packet::add(nndeploy::base::Param &param) { params_.push_back(&param); }

void Packet::add(void *anything) { anything_ = anything; }

// bool Packet::empty() { return emptyBuffer() && emptyMat() && emptyTensor(); }

bool Packet::emptyBuffer() { return buffers_.size() == 0; }
int Packet::getBufferSize() { return buffers_.size(); }
nndeploy::device::Buffer *Packet::getBuffer() { return buffers_[0]; }
nndeploy::device::Buffer *Packet::getBuffer(int index) {
  return buffers_[index];
}

bool Packet::emptyMat() { return mats_.size() == 0; }
int Packet::getMatSize() { return mats_.size(); }
nndeploy::device::Mat *Packet::getMat() { return mats_[0]; }
nndeploy::device::Mat *Packet::getMat(int index) { return mats_[index]; }

bool Packet::emptyTensor() { return tensors_.size() == 0; }
int Packet::getTensorSize() { return tensors_.size(); }
nndeploy::device::Tensor *Packet::getTensor() { return tensors_[0]; }
nndeploy::device::Tensor *Packet::getTensor(int index) {
  return tensors_[index];
}

bool Packet::emptyParam() { return params_.size() == 0; }
int Packet::getParamSize() { return params_.size(); }
nndeploy::base::Param *Packet::getParam() { return params_[0]; }
nndeploy::base::Param *Packet::getParam(int index) { return params_[index]; }

void *Packet::getAnything() { return anything_; }

}  // namespace common
}  // namespace nntask
