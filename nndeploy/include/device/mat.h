/**
 * @file runtime.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_DEVICE_MAT_H_
#define _NNDEPLOY_INCLUDE_DEVICE_MAT_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"

namespace nndeploy {
namespace device {

class Device;

struct MatDesc {
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

class Mat {
 public:
  Mat();
  virtual ~Mat();

  // 暂时未添加stride相关构造函数
  Mat(Device *device, int32_t height, int32_t width, int32_t channel,
      base::DataType data_type);
  Mat(Device *device, int32_t *shape, int32_t shape_len,
      base::DataType data_type);
  Mat(Device *device, const base::IntVector &shape_, base::DataType data_type);
  Mat(Device *device, const MatDesc &desc, const base::IntVector &config);

  Mat(BufferPool *buffer_pool, int32_t height, int32_t width, int32_t channel,
      base::DataType data_type);
  Mat(BufferPool *buffer_pool, int32_t *shape, int32_t shape_len,
      base::DataType data_type);
  Mat(BufferPool *buffer_pool, const base::IntVector &shape_,
      base::DataType data_type);
  Mat(BufferPool *buffer_pool, const MatDesc &desc,
      const base::IntVector &config);

  Mat(const MatDesc &desc, Buffer *buffer);

  //
  Mat(const Mat &mat);
  Mat(Mat &&mat);

  //
  Mat &operator=(const Mat &mat);
  Mat &operator==(Mat &&mat);

  // create
  void create(const MatDesc &desc, Buffer *buffer);
  void create(Device *device, const MatDesc &desc,
              const base::IntVector &config);
  void create(BufferPool *buffer_pool, const MatDesc &desc,
              const base::IntVector &config);

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
  // 必须确保Mat为空
  void create(Device *device, BufferPool *buffer_pool, const MatDesc &desc,
              Buffer *buffer, const base::IntVector &config);

 private:
  MatDesc desc_;
  bool is_external_buffer_ = false;
  Buffer *buffer_ = nullptr;
};

class MatPtrArray {
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