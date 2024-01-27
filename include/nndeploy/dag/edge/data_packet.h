#ifndef _NNDEPLOY_DAG_EDGE_DATA_PACKET_H_
#define _NNDEPLOY_DAG_EDGE_DATA_PACKET_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class DataPacket : public base::NonCopyable {
 public:
  enum Flag : int {
    kFlagBuffer,
    kFlagMat = 1,
#ifdef ENABLE_NNDEPLOY_OPENCV
    kFlagCvMat = 2,
#endif
    kFlagTensor = 4,
    kFlagParam = 8,

    kFlagVoid = 1 << 30,

    kFlagNone = 1 << 31,
  };
  DataPacket();
  virtual ~DataPacket();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual base::Status set(device::Buffer &buffer, int index);
  device::Buffer *create(device::Device *device, const device::BufferDesc &desc,
                         int index);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer();

  virtual base::Status set(device::Mat *mat, int index, bool is_external);
  virtual base::Status set(device::Mat &mat, int index);
  device::Mat *create(device::Device *device, const device::MatDesc &desc,
                      int index, const std::string &name);
  virtual bool notifyWritten(device::Mat *mat);
  virtual device::Mat *getMat();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual base::Status set(cv::Mat &cv_mat, int index);
  virtual cv::Mat *getCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual base::Status set(device::Tensor &tensor, int index);
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         int index, const std::string &name);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor();

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual base::Status set(base::Param &param, int index);
  virtual base::Param *getParam();

  virtual base::Status set(void *anything, int index, bool is_external);
  virtual void *getAnything();

  int getIndex();

 protected:
  void destory();

 protected:
  bool is_external_ = true;
  int index_ = -1;
  Flag flag_ = kFlagNone;
  bool written_ = false;
  void *anything_ = nullptr;
};

class PipelineDataPacket : public DataPacket {
 public:
  PipelineDataPacket(int consumers_size);
  virtual ~PipelineDataPacket();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual base::Status set(device::Buffer &buffer, int index);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer();

  virtual base::Status set(device::Mat *mat, int index, bool is_external);
  virtual base::Status set(device::Mat &mat, int index);
  virtual bool notifyWritten(device::Mat *mat);
  virtual device::Mat *getMat();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual base::Status set(cv::Mat &cv_mat, int index);
  virtual cv::Mat *getCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual base::Status set(device::Tensor &tensor, int index);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor();

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual base::Status set(base::Param &param, int index);
  virtual base::Param *getParam();

  virtual base::Status set(void *anything, int index, bool is_external);
  virtual void *getAnything();

  void increaseConsumersCount();

  int getConsumersSize();
  int getConsumersCount();

 protected:
  std::mutex mutex_;
  std::condition_variable cv_;
  int consumers_size_ = 0;
  int consumers_count_ = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif
