
#ifndef _NNDEPLOY_SOURCE_DEVICE_MAT_H_
#define _NNDEPLOY_SOURCE_DEVICE_MAT_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

struct NNDEPLOY_CC_API MatDesc {
  MatDesc(){};
  explicit MatDesc(base::DataType data_type, const base::IntVector &shape,
                   const base::SizeVector &stride)
      : data_type_(data_type), shape_(shape), stride_(stride){};

  MatDesc(const MatDesc &desc) = default;
  MatDesc &operator=(const MatDesc &desc) = default;

  virtual ~MatDesc(){};

  base::DataType data_type_;

  base::IntVector shape_;

  base::SizeVector stride_;
};

class NNDEPLOY_CC_API Mat {
 public:
  Mat();
  virtual ~Mat();

  Mat(Device *device, const MatDesc &desc,
      const base::IntVector &config = base::IntVector());

  Mat(Device *device, const MatDesc &desc, void *data_ptr,
      const base::IntVector &config = base::IntVector());
  Mat(Device *device, const MatDesc &desc, int32_t data_id,
      const base::IntVector &config = base::IntVector());

  Mat(const MatDesc &desc, Buffer *buffer);

  //
  Mat(const Mat &mat);
  Mat(Mat &&mat);

  //
  Mat &operator=(const Mat &mat);
  Mat &operator==(Mat &&mat);

  // create
  // 必须确保为空
  void create(Device *device, const MatDesc &desc,
              const base::IntVector &config = base::IntVector());

  void create(Device *device, const MatDesc &desc, void *data_ptr,
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const MatDesc &desc, int32_t data_id,
              const base::IntVector &config = base::IntVector());

  void create(const MatDesc &desc, Buffer *buffer);

  // destroy
  void destory();

  // get
  bool empty();
  bool isContinue();

  MatDesc getDesc();
  base::DataType getDataType();
  base::IntVector getShape();
  int32_t getShapeIndex(int index);
  int32_t getHeight();
  int32_t getWidth();
  int32_t getChannel();
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
  int32_t getId();
  BufferSourceType getBufferSourceType();

 private:
  void create(Device *device, const MatDesc &desc, Buffer *buffer,
              void *data_ptr, int32_t data_id, const base::IntVector &config);

 private:
  MatDesc desc_;
  bool is_external_buffer_ = false;
  Buffer *buffer_ = nullptr;
};

class NNDEPLOY_CC_API MatPtrArray {
 public:
  MatPtrArray();
  MatPtrArray(const std::vector<Mat *> &mats);
  MatPtrArray(Mat *mat);
  MatPtrArray(Mat &mat);

  virtual ~MatPtrArray();

  void add(Mat *mat);
  void add(const std::vector<Mat *> &mats);
  void add(Mat &mat);

  bool empty();
  int getSize();
  Mat *get();
  Mat *get(int index);

 private:
  std::vector<Mat *> mats_;
};

}  // namespace device
}  // namespace nndeploy

#endif