
#include "nndeploy/include/device/mat.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

Mat::Mat() {}
Mat::~Mat() { destory(); }
Mat::Mat::Mat(Device *device, int32_t height, int32_t width, int32_t channel,
              base::DataType data_type) {
  desc_.data_type_ = data_type;
  desc_.shape_.push_back(height);
  desc_.shape_.push_back(width);
  desc_.shape_.push_back(channel);
  create(device, desc_, base::IntVector());
}
Mat::Mat(Device *device, int32_t *shape, int32_t shape_len,
         base::DataType data_type) {
  desc_.data_type_ = data_type;
  for (int i = 0; i < shape_len; ++i) {
    desc_.shape_.push_back(shape[i]);
  }
  create(device, desc_, base::IntVector());
}
Mat::Mat(Device *device, const base::IntVector &shape_,
         base::DataType data_type) {
  desc_.data_type_ = data_type;
  desc_.shape_ = shape_;
  create(device, desc_, base::IntVector());
}
Mat::Mat(Device *device, const MatDesc &desc, const base::IntVector &config) {
  create(device, desc_, config);
}
Mat::Mat::Mat(BufferPool *buffer_pool, int32_t height, int32_t width,
              int32_t channel, base::DataType data_type) {
  desc_.data_type_ = data_type;
  desc_.shape_.push_back(height);
  desc_.shape_.push_back(width);
  desc_.shape_.push_back(channel);
  create(buffer_pool, desc_, base::IntVector());
}
Mat::Mat(BufferPool *buffer_pool, int32_t *shape, int32_t shape_len,
         base::DataType data_type) {
  desc_.data_type_ = data_type;
  for (int i = 0; i < shape_len; ++i) {
    desc_.shape_.push_back(shape[i]);
  }
  create(buffer_pool, desc_, base::IntVector());
}
Mat::Mat(BufferPool *buffer_pool, const base::IntVector &shape_,
         base::DataType data_type) {
  desc_.data_type_ = data_type;
  desc_.shape_ = shape_;
  create(buffer_pool, desc_, base::IntVector());
}
Mat::Mat(BufferPool *buffer_pool, const MatDesc &desc,
         const base::IntVector &config) {
  create(buffer_pool, desc, config);
}

Mat::Mat(const MatDesc &desc, Buffer *buffer) { create(desc, buffer); }

//
Mat::Mat(const Mat &mat) {
  if (this == &mat) {
    return;
  }

  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  buffer_->addRef();
}
Mat::Mat(Mat &&mat) {
  if (this == &mat) {
    return;
  }
  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  mat.buffer_ = nullptr;
}

//
Mat &Mat::operator=(const Mat &mat) {
  if (this == &mat) {
    return *this;
  }

  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  buffer_->addRef();
  return *this;
}
Mat &Mat::operator==(Mat &&mat) {
  if (this == &mat) {
    return *this;
  }
  desc_ = mat.desc_;
  buffer_ = mat.buffer_;
  mat.buffer_ = nullptr;
  return *this;
}

// create
void Mat::create(const MatDesc &desc, Buffer *buffer) {
  create(nullptr, nullptr, desc, buffer, base::IntVector());
}
void Mat::create(Device *device, const MatDesc &desc,
                 const base::IntVector &config) {
  create(device, nullptr, desc, nullptr, config);
}
void Mat::create(BufferPool *buffer_pool, const MatDesc &desc,
                 const base::IntVector &config) {
  create(nullptr, buffer_pool, desc, nullptr, config);
}

void Mat::create(Device *device, BufferPool *buffer_pool, const MatDesc &desc,
                 Buffer *buffer, const base::IntVector &config) {
  desc_ = desc;
  if (buffer != nullptr) {
    buffer->addRef();
    buffer_ = buffer;
    return;
  }
  if (buffer_pool != nullptr) {
    BufferDesc buffer_desc =
        buffer_pool->getDevice()->toBufferDesc(desc_, config);
    buffer_ = buffer_pool->malloc(buffer_desc);
    return;
  }
  if (device != nullptr) {
    BufferDesc buffer_desc = device->toBufferDesc(desc, config);
    buffer_ = device->malloc(buffer_desc);
    return;
  }
  return;
}

void Mat::destory() {
  if (buffer_ != nullptr) {
    buffer_->subRef();
    if (buffer_->getRef() == 1) {
      if (buffer_->isBufferPool()) {
        BufferPool *pool = buffer_->getBufferPool();
        pool->free(buffer_);
      } else {
        Device *device = buffer_->getDevice();
        device->free(buffer_);
      }
    }
    buffer_ = nullptr;
  }
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
