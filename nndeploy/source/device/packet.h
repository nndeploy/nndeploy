#ifndef _NNDEPLOY_SOURCE_DEVICE_PACKET_H_
#define _NNDEPLOY_SOURCE_DEVICE_PACKET_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"

namespace nndeploy {
namespace device {

class NNDEPLOY_CC_API Packet {
 public:
  Packet();

  Packet(const std::vector<Buffer *> &buffers);
  Packet(Buffer *buffer);
  Packet(Buffer &buffer);

  Packet(const std::vector<Mat *> &mats);
  Packet(Mat *mat);
  Packet(Mat &mat);

  Packet(const std::vector<Tensor *> &tensors);
  Packet(Tensor *tensor);
  Packet(Tensor &tensor);

  virtual ~Packet();

  void add(const std::vector<Buffer *> &buffers);
  void add(Buffer *buffer);
  void add(Buffer &buffer);

  void add(const std::vector<Mat *> &mats);
  void add(Mat *mat);
  void add(Mat &mat);

  void add(const std::vector<Tensor *> &tensors);
  void add(Tensor *tensor);
  void add(Tensor &tensor);

  bool empty();

  bool emptyBuffer();
  int getBufferSize();
  Buffer *getBuffer();
  Buffer *getBuffer(int index);

  bool emptyMat();
  int getMatSize();
  Mat *getMat();
  Mat *getMat(int index);

  bool emptyTensor();
  int getTensorSize();
  Tensor *getTensor();
  Tensor *getTensor(int index);

 private:
  std::vector<Buffer *> buffers_;
  std::vector<Mat *> mats_;
  std::vector<Tensor *> tensors_;
};

}  // namespace device
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DEVICE_PACKET_H_ */
