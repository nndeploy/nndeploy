#ifndef _NNDEPLOY_SOURCE_TASK_PACKET_H_
#define _NNDEPLOY_SOURCE_TASK_PACKET_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/mat.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/pre_process/params.h"

namespace nndeploy {
namespace task {

class NNDEPLOY_CC_API Packet {
 public:
  Packet();
  Packet(std::string name);

  Packet(const std::vector<device::Buffer *> &buffers);
  Packet(device::Buffer *buffer);
  Packet(device::Buffer &buffer);

  Packet(const std::vector<device::Mat *> &mats);
  Packet(device::Mat *mat);
  Packet(device::Mat &mat);

  Packet(const std::vector<device::Tensor *> &tensors);
  Packet(device::Tensor *tensor);
  Packet(device::Tensor &tensor);

  Packet(const std::vector<base::Param *> &param);
  Packet(base::Param *param);
  Packet(base::Param &param);

#ifdef NNDEPLOY_ENABLE_OPENCV
  Packet(const std::vector<cv::Mat *> &cv_mats);
  Packet(cv::Mat *cv_mat);
  Packet(cv::Mat &cv_mat);
#endif

  virtual ~Packet();

  base::Status setName(std::string name);
  std::string getName();

  void add(const std::vector<device::Buffer *> &buffers);
  void add(device::Buffer *buffer);
  void add(device::Buffer &buffer);

  void add(const std::vector<device::Mat *> &mats);
  void add(device::Mat *mat);
  void add(device::Mat &mat);

  void add(const std::vector<device::Tensor *> &tensors);
  void add(device::Tensor *tensor);
  void add(device::Tensor &tensor);

  void add(const std::vector<base::Param *> &param);
  void add(base::Param *param);
  void add(base::Param &param);

#ifdef NNDEPLOY_ENABLE_OPENCV
  void add(const std::vector<cv::Mat *> &cv_mats);
  void add(cv::Mat *cv_mat);
  void add(cv::Mat &cv_mat);
#endif

  void add(void *anything);

  // bool empty();

  bool emptyBuffer();
  int getBufferSize();
  device::Buffer *getBuffer();
  device::Buffer *getBuffer(int index);

  bool emptyMat();
  int getMatSize();
  device::Mat *getMat();
  device::Mat *getMat(int index);

  bool emptyTensor();
  int getTensorSize();
  device::Tensor *getTensor();
  device::Tensor *getTensor(int index);

  bool emptyParam();
  int getParamSize();
  base::Param *getParam();
  base::Param *getParam(int index);

#ifdef NNDEPLOY_ENABLE_OPENCV
  bool emptyCvMat();
  int getCvMatSize();
  cv::Mat *getCvMat();
  cv::Mat *getCvMat(int index);
#endif

  void *getAnything();

 private:
  std::string name_;

  std::vector<device::Buffer *> buffers_;
  std::vector<device::Mat *> mats_;
  std::vector<device::Tensor *> tensors_;

  std::vector<base::Param *> params_;

#ifdef NNDEPLOY_ENABLE_OPENCV
  std::vector<cv::Mat *> cv_mats_;
#endif

  void *anything_ = nullptr;
};

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_TASK_PACKET_H_ */
