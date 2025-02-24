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
  flag_ = EdgeTypeFlag::kBuffer;
  written_ = true;
  anything_ = (void *)buffer;
  deleter_ = [](void *d) { delete static_cast<device::Buffer *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(device::Buffer));
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
    if (flag_ != EdgeTypeFlag::kBuffer) {
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
  flag_ = EdgeTypeFlag::kBuffer;
  written_ = false;
  anything_ = (void *)(buffer);
  deleter_ = [](void *d) { delete static_cast<device::Buffer *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(device::Buffer));
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
  if (flag_ != EdgeTypeFlag::kBuffer) {
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
  flag_ = EdgeTypeFlag::kCvMat;
  written_ = true;
  anything_ = (void *)cv_mat;
  deleter_ = [](void *d) { delete static_cast<cv::Mat *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(cv::Mat));
  return status;
}
cv::Mat *DataPacket::create(int rows, int cols, int type,
                            const cv::Vec3b &value, int index) {
  base::Status status = base::kStatusCodeOk;
  cv::Mat *cv_mat = nullptr;
  if (anything_ == nullptr) {
    cv_mat = new cv::Mat(rows, cols, type, value);
  } else {
    if (flag_ != EdgeTypeFlag::kCvMat) {
      destory();
      cv_mat = new cv::Mat(rows, cols, type, value);
    } else {
      cv_mat = (cv::Mat *)(anything_);
      if (cv_mat->rows != rows || cv_mat->cols != cols ||
          cv_mat->type() != type) {
        destory();
        cv_mat = new cv::Mat(rows, cols, type, value);
      }
    }
  }
  is_external_ = false;
  index_ = index;
  flag_ = EdgeTypeFlag::kCvMat;
  written_ = false;
  anything_ = (void *)(cv_mat);
  deleter_ = [](void *d) { delete static_cast<cv::Mat *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(cv::Mat));
  return cv_mat;
}
bool DataPacket::notifyWritten(cv::Mat *cv_mat) {
  if ((void *)cv_mat == anything_) {
    written_ = true;
    return true;
  } else {
    return false;
  }
}
cv::Mat *DataPacket::getCvMat() {
  if (flag_ != EdgeTypeFlag::kCvMat) {
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
  flag_ = EdgeTypeFlag::kTensor;
  written_ = true;
  anything_ = (void *)tensor;
  deleter_ = [](void *d) { delete static_cast<device::Tensor *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(device::Tensor));
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
    if (flag_ != EdgeTypeFlag::kTensor) {
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
  flag_ = EdgeTypeFlag::kTensor;
  written_ = false;
  anything_ = (void *)(tensor);
  deleter_ = [](void *d) { delete static_cast<device::Tensor *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(device::Tensor));
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
  if (flag_ != EdgeTypeFlag::kTensor) {
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
  flag_ = EdgeTypeFlag::kParam;
  written_ = true;
  anything_ = (void *)param;
  deleter_ = [](void *d) { delete static_cast<base::Param *>(d); };
  type_info_ = const_cast<std::type_info *>(&typeid(base::Param));
  return status;
}
base::Param *DataPacket::getParam() {
  if (flag_ != EdgeTypeFlag::kParam) {
    return nullptr;
  } else {
    return (base::Param *)(anything_);
  }
}
bool DataPacket::notifyWritten(base::Param *param) {
  if ((void *)param == anything_) {
    written_ = true;
    return true;
  } else {
    return false;
  }
}

base::Status DataPacket::takeDataPacket(DataPacket *packet) {
  base::Status status = base::kStatusCodeOk;
  if (packet == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }
  if (anything_ != nullptr) {
    destory();
  }
  is_external_ = packet->is_external_;
  index_ = packet->index_;
  flag_ = packet->flag_;
  written_ = packet->written_;
  anything_ = packet->anything_;
  type_info_ = packet->type_info_;
  deleter_ = packet->deleter_;

  packet->is_external_ = true;
  packet->index_ = -1;
  packet->flag_ = EdgeTypeFlag::kNone;
  packet->written_ = false;
  packet->anything_ = nullptr;
  packet->type_info_ = nullptr;
  packet->deleter_ = nullptr;
  delete packet;
  packet = nullptr;

  return status;
}

int DataPacket::getIndex() { return index_; }

void DataPacket::destory() {
  if (!is_external_ && anything_ != nullptr) {
#if 1
    if (flag_ == EdgeTypeFlag::kBuffer) {
      device::Buffer *tmp = (device::Buffer *)(anything_);
      delete tmp;
    }
#ifdef ENABLE_NNDEPLOY_OPENCV
    else if (flag_ == EdgeTypeFlag::kCvMat) {
      cv::Mat *tmp = (cv::Mat *)(anything_);
      delete tmp;
    }
#endif
    else if (flag_ == EdgeTypeFlag::kTensor) {
      device::Tensor *tmp = (device::Tensor *)(anything_);
      delete tmp;
    } else if (flag_ == EdgeTypeFlag::kParam) {
      base::Param *tmp = (base::Param *)(anything_);
      delete tmp;
    } else if (flag_ == EdgeTypeFlag::kAny) {
      deleter_(anything_);
    }
#else
    deleter_(anything_);
#endif
  }

  is_external_ = true;
  index_ = -1;
  flag_ = EdgeTypeFlag::kNone;
  written_ = false;
  anything_ = nullptr;
  type_info_ = nullptr;
  deleter_ = nullptr;
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
bool PipelineDataPacket::notifyWritten(cv::Mat *cv_mat) {
  std::unique_lock<std::mutex> lock(mutex_);
  bool status = DataPacket::notifyWritten(cv_mat);
  if (status) {
    cv_.notify_all();
  }

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
bool PipelineDataPacket::notifyWritten(base::Param *param) {
  std::unique_lock<std::mutex> lock(mutex_);
  bool status = DataPacket::notifyWritten(param);
  if (status) {
    cv_.notify_all();
  }
  return status;
}
base::Param *PipelineDataPacket::getParam() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return written_; });
  return DataPacket::getParam();
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