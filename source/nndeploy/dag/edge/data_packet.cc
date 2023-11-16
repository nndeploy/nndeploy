#include "nndeploy/dag/edge/abstract_edge.h"

namespace nndeploy {
namespace dag {

DataPacket::DataPacket() {}

DataPacket::~DataPacket() {
  if (!is_external_) {
    if (flag_ == kFlagBuffer) {
      device::Buffer* tmp = (device::Buffer*)(anything_);
      destory(buffer);
    } else if (flag_ == kFlagMat) {
      device::Mat* tmp = (device::Mat*)(anything_);
      delete tmp;
    }
#ifdef ENABLE_NNDEPLOY_OPENCV
    else if (flag_ == kFlagCvMat) {
      cv::Mat* tmp = (cv::Mat*)(anything_);
      delete tmp;
    }
#endif
    else if (flag_ == kFlagTensor) {
      device::Tensor* tmp = (device::Tensor*)(anything_);
      delete tmp;
    } else if (flag_ == kFlagParam) {
      base::Param* tmp = (base::Param*)(anything_);
      delete tmp;
    }
  }
}

}  // namespace dag
}  // namespace nndeploy