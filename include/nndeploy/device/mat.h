
#ifndef _NNDEPLOY_DEVICE_MAT_H_
#define _NNDEPLOY_DEVICE_MAT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

struct NNDEPLOY_CC_API MatDesc {
  MatDesc(){};
  explicit MatDesc(base::DataType data_type, const base::IntVector &shape,
                   const base::SizeVector &stride)
      : data_type_(data_type), shape_(shape), stride_(stride){};

  MatDesc(const MatDesc &desc) {
    data_type_ = desc.data_type_;
    shape_ = desc.shape_;
    stride_ = desc.stride_;
  }
  MatDesc &operator=(const MatDesc &desc) {
    data_type_ = desc.data_type_;
    shape_ = desc.shape_;
    stride_ = desc.stride_;
    return *this;
  }

  MatDesc(MatDesc &&desc) {
    data_type_ = desc.data_type_;
    shape_ = std::move(desc.shape_);
    stride_ = std::move(desc.stride_);
  }
  MatDesc &operator=(MatDesc &&desc) {
    data_type_ = desc.data_type_;
    shape_ = std::move(desc.shape_);
    stride_ = std::move(desc.stride_);
    return *this;
  }

  virtual ~MatDesc(){};

  base::DataType data_type_;

  base::IntVector shape_;

  base::SizeVector stride_;
};

class NNDEPLOY_CC_API Mat {
 public:
  Mat();
  virtual ~Mat();

  Mat(const std::string &name);

  Mat(const MatDesc &desc, const std::string &name = "");

  Mat(Device *device, const MatDesc &desc, const std::string &name = "",
      const base::IntVector &config = base::IntVector());

  Mat(Device *device, const MatDesc &desc, void *data_ptr,
      const std::string &name = "",
      const base::IntVector &config = base::IntVector());
  Mat(Device *device, const MatDesc &desc, int data_id,
      const std::string &name = "",
      const base::IntVector &config = base::IntVector());

  Mat(const MatDesc &desc, Buffer *buffer, const std::string &name = "");

  //
  Mat(const Mat &mat);
  Mat(Mat &&mat);

  //
  Mat &operator=(const Mat &mat);
  Mat &operator==(Mat &&mat);

  // create
  // 必须确保为空
  void create(const MatDesc &desc, const std::string &name = "");

  void create(Device *device, const MatDesc &desc, const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(Device *device, const MatDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const MatDesc &desc, int data_id,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(const MatDesc &desc, Buffer *buffer,
              const std::string &name = "");

  // destroy
  void destory();

  // alloc
  void allocBuffer(Device *device,
                   const base::IntVector &config = base::IntVector());
  void deallocateBuffer();

  // modify
  bool justModify(const MatDesc &desc);
  bool justModify(Buffer *buffer);

  // get
  bool empty();
  bool isContinue();
  bool isExternalBuffer();

  std::string getName();

  MatDesc getDesc();
  base::DataType getDataType();
  base::IntVector getShape();
  int getShapeIndex(int index);
  int getHeight();
  int getWidth();
  int getChannel();
  base::SizeVector getStride();
  size_t getStrideIndex(int index);

  Buffer *getBuffer();
  base::DeviceType getDeviceType();
  Device *getDevice();
  BufferPool *getBufferPool();
  bool isBufferPool();
  BufferDesc getBufferDesc();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int getId();
  BufferSourceType getBufferSourceType();

 private:
  void create(Device *device, const MatDesc &desc, Buffer *buffer,
              void *data_ptr, int data_id, const std::string &name,
              const base::IntVector &config);

 private:
  std::string name_;
  MatDesc desc_;
  bool is_external_buffer_ = false;
  Buffer *buffer_ = nullptr;
};

}  // namespace device
}  // namespace nndeploy

#endif