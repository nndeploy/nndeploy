#include "nndeploy/dag/edge/data_packet.h"

namespace nndeploy {
namespace dag {

DataPacket::DataPacket() {}

DataPacket::~DataPacket() { destory(); }

base::Status DataPacket::set(device::Buffer *buffer, int index,
                             bool is_external) {
  base::Status status = base::kStatusCodeOk;
  if (buffer != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagBuffer;
  written_ = true;
  anything_ = (void *)buffer;
  return status;
}
base::Status DataPacket::set(device::Buffer &buffer, int index) {
  base::Status status = base::kStatusCodeOk;
  if (&buffer != anything_) {
    destory();
  }
  is_external_ = true;
  index_ = index;
  flag_ = kFlagBuffer;
  written_ = true;
  anything_ = (void *)(&buffer);
  return status;
}
device::Buffer *DataPacket::create(device::Device *device,
                                   const device::BufferDesc &desc, int index) {
  base::Status status = base::kStatusCodeOk;
  device::Buffer *buffer = nullptr;
  if (anything_ == nullptr) {
    void *data = device->allocate(desc);
    buffer = new device::Buffer(device, desc, data);
  } else {
    if (flag_ != kFlagBuffer) {
      destory();
      void *data = device->allocate(desc);
      buffer = new device::Buffer(device, desc, data);
    } else {
      buffer = (device::Buffer *)(anything_);
      if (buffer->getDesc() != desc) {
        destory();
        void *data = device->allocate(desc);
        buffer = new device::Buffer(device, desc, data);
      }
    }
  }
  is_external_ = false;
  index_ = index;
  flag_ = kFlagBuffer;
  written_ = false;
  anything_ = (void *)(buffer);
  return buffer;
}
bool DataPacket::notifyWritten(device::Buffer *buffer) {
  if ((void *)buffer == anything_) {
    written_ = true;
    return true;
  } else {
    return false;
  }
}
device::Buffer *DataPacket::getBuffer() {
  if (flag_ != kFlagBuffer) {
    return nullptr;
  } else {
    return (device::Buffer *)(anything_);
  }
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status DataPacket::set(cv::Mat *cv_mat, int index, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  if (cv_mat != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagCvMat;
  written_ = true;
  anything_ = (void *)cv_mat;
  return status;
}
base::Status DataPacket::set(cv::Mat &cv_mat, int index) {
  base::Status status = base::kStatusCodeOk;
  if (&cv_mat != anything_) {
    destory();
  }
  is_external_ = true;
  index_ = index;
  flag_ = kFlagCvMat;
  written_ = true;
  anything_ = (void *)(&cv_mat);
  return status;
}
cv::Mat *DataPacket::getCvMat() {
  if (flag_ != kFlagCvMat) {
    return nullptr;
  } else {
    return (cv::Mat *)(anything_);
  }
}
#endif

base::Status DataPacket::set(device::Tensor *tensor, int index,
                             bool is_external) {
  base::Status status = base::kStatusCodeOk;
  if (tensor != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagTensor;
  written_ = true;
  anything_ = (void *)tensor;
  return status;
}
base::Status DataPacket::set(device::Tensor &tensor, int index) {
  base::Status status = base::kStatusCodeOk;
  if (&tensor != anything_) {
    destory();
  }
  is_external_ = true;
  index_ = index;
  flag_ = kFlagTensor;
  written_ = true;
  anything_ = (void *)(&tensor);
  return status;
}
device::Tensor *DataPacket::create(device::Device *device,
                                   const device::TensorDesc &desc, int index,
                                   const std::string &name) {
  base::Status status = base::kStatusCodeOk;
  device::Tensor *tensor = nullptr;
  if (anything_ == nullptr) {
    tensor = new device::Tensor(device, desc, name);
  } else {
    if (flag_ != kFlagTensor) {
      destory();
      tensor = new device::Tensor(device, desc, name);
    } else {
      tensor = (device::Tensor *)(anything_);
      if (tensor->getDesc() != desc) {
        destory();
        tensor = new device::Tensor(device, desc, name);
      }
    }
  }
  is_external_ = false;
  index_ = index;
  flag_ = kFlagTensor;
  written_ = false;
  anything_ = (void *)(tensor);
  return tensor;
}
bool DataPacket::notifyWritten(device::Tensor *tensor) {
  if ((void *)tensor == anything_) {
    written_ = true;
    return true;
  } else {
    return false;
  }
}
device::Tensor *DataPacket::getTensor() {
  if (flag_ != kFlagTensor) {
    return nullptr;
  } else {
    return (device::Tensor *)(anything_);
  }
}

base::Status DataPacket::set(base::Param *param, int index, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  if (param != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagParam;
  written_ = true;
  anything_ = (void *)param;
  return status;
}
base::Status DataPacket::set(base::Param &param, int index) {
  base::Status status = base::kStatusCodeOk;
  if (&param != anything_) {
    destory();
  }
  is_external_ = true;
  index_ = index;
  flag_ = kFlagParam;
  written_ = true;
  anything_ = (void *)(&param);
  return status;
}
base::Param *DataPacket::getParam() {
  if (flag_ != kFlagParam) {
    return nullptr;
  } else {
    return (base::Param *)(anything_);
  }
}

base::Status DataPacket::set(void *anything, int index, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  if (anything != anything_) {
    destory();
  }
  is_external_ = is_external;
  index_ = index;
  flag_ = kFlagVoid;
  written_ = true;
  anything_ = anything;
  return status;
}
void *DataPacket::getAnything() {
  if (flag_ != kFlagVoid) {
    return nullptr;
  } else {
    return (anything_);
  }
}

int DataPacket::getIndex() { return index_; }

void DataPacket::destory() {
  if (!is_external_ && anything_ != nullptr) {
    if (flag_ == kFlagBuffer) {
      device::Buffer *tmp = (device::Buffer *)(anything_);
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

  is_external_ = true;
  index_ = -1;
  flag_ = kFlagNone;
  written_ = false;
  anything_ = nullptr;
}

PipelineDataPacket::PipelineDataPacket(int consumers_size)
    : DataPacket(), consumers_size_(consumers_size), consumers_count_(0) {}

PipelineDataPacket::~PipelineDataPacket() {
  consumers_size_ = 0;
  consumers_count_ = 0;
}

base::Status PipelineDataPacket::set(device::Buffer *buffer, int index,
                                     bool is_external) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(buffer, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
base::Status PipelineDataPacket::set(device::Buffer &buffer, int index) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(buffer, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
bool PipelineDataPacket::notifyWritten(device::Buffer *buffer) {
  std::unique_lock<std::mutex> lock(mutex_);
  bool status = DataPacket::notifyWritten(buffer);
  if (status) {
    cv_.notify_all();
  }

  return status;
}
device::Buffer *PipelineDataPacket::getBuffer() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return written_; });
  return DataPacket::getBuffer();
}

#ifdef ENABLE_NNDEPLOY_OPENCV
base::Status PipelineDataPacket::set(cv::Mat *cv_mat, int index,
                                     bool is_external) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(cv_mat, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
base::Status PipelineDataPacket::set(cv::Mat &cv_mat, int index) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(cv_mat, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
cv::Mat *PipelineDataPacket::getCvMat() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return written_; });
  return DataPacket::getCvMat();
}
#endif

base::Status PipelineDataPacket::set(device::Tensor *tensor, int index,
                                     bool is_external) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(tensor, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
base::Status PipelineDataPacket::set(device::Tensor &tensor, int index) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(tensor, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
bool PipelineDataPacket::notifyWritten(device::Tensor *tensor) {
  std::unique_lock<std::mutex> lock(mutex_);
  bool status = DataPacket::notifyWritten(tensor);
  if (status) {
    cv_.notify_all();
  }
  return status;
}
device::Tensor *PipelineDataPacket::getTensor() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return written_; });
  return DataPacket::getTensor();
}

base::Status PipelineDataPacket::set(base::Param *param, int index,
                                     bool is_external) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(param, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
base::Status PipelineDataPacket::set(base::Param &param, int index) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(param, index);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
base::Param *PipelineDataPacket::getParam() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return written_; });
  return DataPacket::getParam();
}

base::Status PipelineDataPacket::set(void *anything, int index,
                                     bool is_external) {
  std::unique_lock<std::mutex> lock(mutex_);
  base::Status status = DataPacket::set(anything, index, is_external);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "DataPacket::set failed!\n");
  cv_.notify_all();
  return status;
}
void *PipelineDataPacket::getAnything() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return written_; });
  return DataPacket::getAnything();
}

void PipelineDataPacket::increaseConsumersSize() {
  std::unique_lock<std::mutex> lock(mutex_);
  consumers_size_++;
}
void PipelineDataPacket::increaseConsumersCount() {
  std::unique_lock<std::mutex> lock(mutex_);
  consumers_count_++;
}
int PipelineDataPacket::getConsumersSize() { return consumers_size_; }
int PipelineDataPacket::getConsumersCount() { return consumers_count_; }

}  // namespace dag
}  // namespace nndeploy