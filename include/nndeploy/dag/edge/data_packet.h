#ifndef _NNDEPLOY_DAG_EDGE_EDGE_DATA_PACKET_H_
#define _NNDEPLOY_DAG_EDGE_EDGE_DATA_PACKET_H_

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

  base::Status set(device::Buffer *buffer, int index_ = -1,
                   bool is_external = true);
  base::Status set(device::Buffer &buffer, int index_ = -1,
                   bool is_external = true);
  base::Status create(device::Device *device, const device::BufferDesc &desc,
                      int index_ = -1);
  device::Buffer *getBuffer();

  base::Status set(device::Mat *mat, int index_ = -1, bool is_external = true);
  base::Status set(device::Mat &mat, int index_ = -1, bool is_external = true);
  base::Status create(device::Device *device, const device::MatDesc &desc,
                      int index_ = -1);
  device::Mat *getMat();

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, int index_ = -1, bool is_external = true);
  base::Status set(cv::Mat &cv_mat, int index_ = -1, bool is_external = true);
  cv::Mat *getCvMat();
#endif

  base::Status set(device::Tensor *tensor, int index_ = -1,
                   bool is_external = true);
  base::Status set(device::Tensor &tensor, int index_ = -1,
                   bool is_external = true);
  base::Status create(device::Device *device, const device::TensorDesc &desc,
                      int index_ = -1);
  device::Tensor *getTensor();

  base::Status set(base::Param *param, bool is_external = true, int pts = -1);
  base::Status set(base::Param &param, bool is_external = true, int pts = -1);
  base::Param *getParam();

  base::Status set(void *anything, bool is_external = true, int pts = -1);
  void *getAnything();

  int getIndex();

 private:
  bool is_external_ = true;
  int index_ = -1;
  Flag flag_ = kFlagNone;
  void *anything_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* E89A600B_4DA6_4278_B3D4_5AEBE24E8745 */
