
#include "nndeploy/source/device/packet.h"

namespace nndeploy {
namespace device {

Packet::Packet() {}

Packet::Packet(const std::vector<Buffer *> &buffers) : buffers_(buffers) {}
Packet::Packet(Buffer *buffer) { buffers_.push_back(buffer); }
Packet::Packet(Buffer &buffer) { buffers_.push_back(&buffer); }


Packet::Packet(const std::vector<Mat *> &mats) : mats_(mats) {}
Packet::Packet(Mat *mat) { mats_.push_back(mat); }
Packet::Packet(Mat &mat) { mats_.push_back(&mat); }

Packet::Packet(const std::vector<Tensor *> &tensors)
    : tensors_(tensors) {}
Packet::Packet(Tensor *tensor) { tensors_.push_back(tensor); }
Packet::Packet(Tensor &tensor) { tensors_.push_back(&tensor); }

Packet::~Packet() {}

void Packet::add(Buffer *buffer) { buffers_.push_back(buffer); }
void Packet::add(const std::vector<Buffer *> &buffers) {
  for (auto buffer : buffers) {
    buffers_.push_back(buffer);
  }
}
void Packet::add(Buffer &buffer) { buffers_.push_back(&buffer); }

void Packet::add(Mat *mat) { mats_.push_back(mat); }
void Packet::add(const std::vector<Mat *> &mats) {
  for (auto mat : mats) {
    mats_.push_back(mat);
  }
}
void Packet::add(Mat &mat) { mats_.push_back(&mat); }

void Packet::add(Tensor *tensor) { tensors_.push_back(tensor); }
void Packet::add(const std::vector<Tensor *> &tensors) {
  for (auto tensor : tensors) {
    tensors_.push_back(tensor);
  }
}
void Packet::add(Tensor &tensor) { tensors_.push_back(&tensor); }

bool Packet::empty() { return emptyBuffer() && emptyMat() && emptyTensor(); }

bool Packet::emptyBuffer() { return buffers_.size() == 0; }
int Packet::getBufferSize() { return buffers_.size(); }
Buffer *Packet::getBuffer() { return buffers_[0]; }
Buffer *Packet::getBuffer(int index) { return buffers_[index]; }

bool Packet::emptyMat() { return mats_.size() == 0; }
int Packet::getMatSize() { return mats_.size(); }
Mat *Packet::getMat() { return mats_[0]; }
Mat *Packet::getMat(int index) { return mats_[index]; }


bool Packet::emptyTensor() { return tensors_.size() == 0; }
int Packet::getTensorSize() { return tensors_.size(); }
Tensor *Packet::getTensor() { return tensors_[0]; }
Tensor *Packet::getTensor(int index) { return tensors_[index]; }

}  // namespace device
}  // namespace nndeploy

