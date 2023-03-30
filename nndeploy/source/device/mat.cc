
#include "nndeploy/source/device/mat.h"

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

Mat::Mat() {}
Mat::~Mat() { destory(); }

Mat::Mat(Device *device, const MatDesc &desc, const base::IntVector &config) {
  create(device, desc, config);
}

Mat::Mat(Device *device, const MatDesc &desc, void *data_ptr,
         const base::IntVector &config ) {
  create(device, desc, data_ptr, config);
}
Mat::Mat(Device *device, const MatDesc &desc, int32_t data_id,
         const base::IntVector &config ) {
  create(device, desc, data_id, config);
}

Mat::Mat(const MatDesc &desc, Buffer *buffer) { create(desc, buffer); }

//
Mat::Mat(const Mat &mat) {
  if (this == &mat) {
    return;
  }

  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  if (buffer_ != nullptr && is_external_buffer_ == false) {
    buffer_->addRef();
  }
}
Mat::Mat(Mat &&mat) {
  if (this == &mat) {
    return;
  }
  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  is_external_buffer_ = mat.is_external_buffer_;
  mat.buffer_ = nullptr;
}

//
Mat &Mat::operator=(const Mat &mat) {
  if (this == &mat) {
    return *this;
  }

  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  if (buffer_ != nullptr && is_external_buffer_ == false) {
    buffer_->addRef();
  }
  return *this;
}
Mat &Mat::operator==(Mat &&mat) {
  if (this == &mat) {
    return *this;
  }
  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  is_external_buffer_ = mat.is_external_buffer_;
  mat.buffer_ = nullptr;
  return *this;
}

// create
// 必须确保为空
void Mat::create(Device *device, const MatDesc &desc,
                 const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, -1, config);
}

void Mat::create(Device *device, const MatDesc &desc, void *data_ptr,
                 const base::IntVector &config) {
  create(device, desc, nullptr, data_ptr, -1, config);
}
void Mat::create(Device *device, const MatDesc &desc, int32_t data_id,
                 const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, data_id, config);
}

void Mat::create(const MatDesc &desc, Buffer *buffer) {
  create(nullptr, desc, buffer, nullptr, -1, base::IntVector());
}

void Mat::create(Device *device, const MatDesc &desc, Buffer *buffer,
                 void *data_ptr, int32_t data_id,
                 const base::IntVector &config) {
  desc_ = desc;
  if (buffer != nullptr) {
    is_external_buffer_ = true;
    buffer_ = buffer;
    return;
  }
  if (device != nullptr) {
    is_external_buffer_ = false;
    if (data_ptr != nullptr) {
      BufferDesc buffer_desc = device->toBufferDesc(desc, config);
      buffer_ =
          device->create(buffer_desc, data_ptr, kBufferSourceTypeExternal);
      return;
    } else if (data_id != -1) {
      BufferDesc buffer_desc = device->toBufferDesc(desc, config);
      buffer_ = device->create(buffer_desc, data_id, kBufferSourceTypeExternal);
      return;
    } else {
      BufferDesc buffer_desc = device->toBufferDesc(desc, config);
      buffer_ = device->allocate(buffer_desc);
      return;
    }
  }
  return;
}

void Mat::destory() {
  desc_.data_type_ = base::DataTypeOf<float>();
  desc_.shape_.clear();
  desc_.stride_.clear();

  is_external_buffer_ = false;

  if (buffer_ != nullptr && is_external_buffer_ == false) {
    buffer_->subRef();
    if (buffer_->getRef() == 0) {
      Device *device = buffer_->getDevice();
      device->deallocate(buffer_);
    }
  }
  buffer_ = nullptr;
}

// get
bool Mat::empty() { return buffer_->empty(); }
bool Mat::isContinue() {
  if (desc_.stride_.size() == 0) {
    return true;
  } else {
    int size = desc_.stride_.size();
    size_t acc = 1;
    for (int i = size - 1; i >= 0; --i) {
      acc *= desc_.shape_[i];
      if (desc_.stride_[i] != acc) {
        return false;
      }
    }
  }
  return true;
}

MatDesc Mat::getDesc() { return desc_; }
base::DataType Mat::getDataType() { return desc_.data_type_; }
base::IntVector Mat::getShape() { return desc_.shape_; }
int32_t Mat::getShapeIndex(int index) { return desc_.shape_[index]; }
int32_t Mat::getHeight() { return desc_.shape_[0]; }
int32_t Mat::getWidth() { return desc_.shape_[1]; }
int32_t Mat::getChannel() { return desc_.shape_[2]; }
base::SizeVector Mat::getStride() { return desc_.stride_; }
size_t Mat::getStrideIndex(int index) { return desc_.stride_[index]; }

Buffer *Mat::getBuffer() { return buffer_; }
base::DeviceType Mat::getDeviceType() { return buffer_->getDeviceType(); }
Device *Mat::getDevice() { return buffer_->getDevice(); }
BufferPool *Mat::getBufferPool() { return buffer_->getBufferPool(); }
bool Mat::isBufferPool() { return buffer_->isBufferPool(); }
BufferDesc Mat::getBufferDesc() { return buffer_->getDesc(); }
size_t Mat::getSize() { return buffer_->getSize(); }
base::SizeVector Mat::getSizeVector() { return buffer_->getSizeVector(); }
base::IntVector Mat::getConfig() { return buffer_->getConfig(); }
void *Mat::getPtr() { return buffer_->getPtr(); }
int32_t Mat::getId() { return buffer_->getId(); }
BufferSourceType Mat::getBufferSourceType() {
  return buffer_->getBufferSourceType();
}

MatPtrArray::MatPtrArray() {}
MatPtrArray::MatPtrArray(const std::vector<Mat *> &mats) : mats_(mats) {}
MatPtrArray::MatPtrArray(Mat *mat) { mats_.push_back(mat); }
MatPtrArray::MatPtrArray(Mat &mat) { mats_.push_back(&mat); }

MatPtrArray::~MatPtrArray() {}

void MatPtrArray::add(Mat *mat) { mats_.push_back(mat); }
void MatPtrArray::add(const std::vector<Mat *> &mats) {
  for (auto mat : mats) {
    mats_.push_back(mat);
  }
}
void MatPtrArray::add(Mat &mat) { mats_.push_back(&mat); }

bool MatPtrArray::empty() { return mats_.size() == 0; }
int MatPtrArray::getSize() { return mats_.size(); }
Mat *MatPtrArray::get() { return mats_[0]; }
Mat *MatPtrArray::get(int index) { return mats_[index]; }

}  // namespace device
}  // namespace nndeploy
