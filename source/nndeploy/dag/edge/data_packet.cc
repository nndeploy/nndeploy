#include "nndeploy/dag/edge/abstract_edge.h"

namespace nndeploy {
namespace dag {

DataPacket::DataPacket() {}

DataPacket::~DataPacket() { destory(); }

base::Status DataPacket::set(device::Buffer *buffer, int index,
                             bool is_external) {
  if (buffer != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagBuffer;
  anything_ = (void *)buffer;
}
base::Status DataPacket::set(device::Buffer &buffer, int index,
                             bool is_external) {
  if (&buffer != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagBuffer;
  anything_ = (void *)(&buffer);
}
base::Status DataPacket::create(device::Device *device,
                                const device::BufferDesc &desc, int index) {
  device::Buffer *buffer = nullptr;
  if (anything_ == nullptr) {
    buffer = device->allocate(desc);
  } else {
    if (flag_ != kFlagBuffer) {
      destory();
      buffer = device->allocate(desc);
    } else {
      buffer = (device::Buffer *)(anything_);
      if (buffer->desc() != desc) {
        destory();
        buffer = device->allocate(desc);
      }
    }
  }
  is_external_ = false;
  index_ = index;
  flag_ = kFlagBuffer;
  anything_ = (void *)(&buffer);
}
device::Buffer *DataPacket::getBuffer() {
  return (device::Buffer *)(anything_);
}

base::Status DataPacket::set(device::Mat *mat, int index, bool is_external);
base::Status DataPacket::set(device::Mat &mat, int index, bool is_external);
base::Status DataPacket::create(device::Device *device,
                                const device::MatDesc &desc, int index);
device::Mat *getMat();

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status DataPacket::set(cv::Mat *cv_mat, int index, bool is_external);
base::Status DataPacket::set(cv::Mat &cv_mat, int index, bool is_external);
cv::Mat *getCvMat();
#endif

base::Status DataPacket::set(device::Tensor *tensor, int index,
                             bool is_external);
base::Status DataPacket::set(device::Tensor &tensor, int index,
                             bool is_external);
base::Status DataPacket::create(device::Device *device,
                                const device::TensorDesc &desc, int index);
device::Tensor *getTensor();

base::Status DataPacket::set(base::Param *param, int index, bool is_external);
base::Status DataPacket::set(base::Param &param, int index, bool is_external);
base::Param *getParam();

base::Status DataPacket::set(void *anything, int index, bool is_external);
void *getAnything();

int getIndex();

void DataPacket::destory() {
  if (!is_external_ && anything_ != nullptr) {
    if (flag_ == kFlagBuffer) {
      device::Buffer *tmp = (device::Buffer *)(anything_);
      destory(buffer);
    } else if (flag_ == kFlagMat) {
      device::Mat *tmp = (device::Mat *)(anything_);
      delete tmp;
    }
#ifdef ENABLE_NNDEPLOY_OPENCV
    else if (flag_ == kFlagCvMat) {
      cv::Mat *tmp = (cv::Mat *)(anything_);
      delete tmp;
    }
#endif
    else if (flag_ == kFlagTensor) {
      device::Tensor *tmp = (device::Tensor *)(anything_);
      delete tmp;
    } else if (flag_ == kFlagParam) {
      base::Param *tmp = (base::Param *)(anything_);
      delete tmp;
    }
  }
}

}  // namespace dag
}  // namespace nndeploy