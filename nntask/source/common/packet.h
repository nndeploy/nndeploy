#ifndef __NNTASK_SOURCE_COMMON_PACKET_H_
#define __NNTASK_SOURCE_COMMON_PACKET_H_

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
#include "nntask/source/common/params.h"

namespace nntask {
namespace common {

class NNDEPLOY_CC_API Packet {
 public:
  Packet();
  Packet(std::string name);

  Packet(const std::vector<nndeploy::device::Buffer *> &buffers);
  Packet(nndeploy::device::Buffer *buffer);
  Packet(nndeploy::device::Buffer &buffer);

  Packet(const std::vector<nndeploy::device::Mat *> &mats);
  Packet(nndeploy::device::Mat *mat);
  Packet(nndeploy::device::Mat &mat);

  Packet(const std::vector<nndeploy::device::Tensor *> &tensors);
  Packet(nndeploy::device::Tensor *tensor);
  Packet(nndeploy::device::Tensor &tensor);

  packet(const std::vector<nndeploy::base::Param *> &param);
  packet(nndeploy::base::Param *param);
  packet(nndeploy::base::Param &param);

#ifdef NNTASK_ENABLE_OPENCV
  Packet(const std::vector<cv::mat *> &cv_mats);
  Packet(cv::mat *cv_mat);
  Packet(cv::mat &cv_mat);
#endif

  virtual ~Packet();

  nndeploy::base::Status setName(std::string name);
  std::string getName();

  void add(const std::vector<nndeploy::device::Buffer *> &buffers);
  void add(nndeploy::device::Buffer *buffer);
  void add(nndeploy::device::Buffer &buffer);

  void add(const std::vector<nndeploy::device::Mat *> &mats);
  void add(nndeploy::device::Mat *mat);
  void add(nndeploy::device::Mat &mat);

  void add(const std::vector<nndeploy::device::Tensor *> &tensors);
  void add(nndeploy::device::Tensor *tensor);
  void add(nndeploy::device::Tensor &tensor);

  void add(const std::vector<nndeploy::base::Param *> &param);
  void add(nndeploy::base::Param *param);
  void add(nndeploy::base::Param &param);

#ifdef NNTASK_ENABLE_OPENCV
  Void add(const std::vector<cv::mat *> &cv_mats);
  Void add(cv::mat *cv_mat);
  Void add(cv::mat &cv_mat);
#endif

  void add(void *anything);

  // bool empty();

  bool emptyBuffer();
  int getBufferSize();
  nndeploy::device::Buffer *getBuffer();
  nndeploy::device::Buffer *getBuffer(int index);

  bool emptyMat();
  int getMatSize();
  nndeploy::device::Mat *getMat();
  nndeploy::device::Mat *getMat(int index);

  bool emptyTensor();
  int getTensorSize();
  nndeploy::device::Tensor *getTensor();
  nndeploy::device::Tensor *getTensor(int index);

  bool emptyParam();
  int getParamSize();
  nndeploy::base::Param *getParam();
  nndeploy::base::Param *getParam(int index);

#ifdef NNTASK_ENABLE_OPENCV
  bool emptyCvMat();
  int getCvMatSize();
  cv::mat *getCvMat();
  cv::mat *getCvMat(int index);
#endif

  void *getAnything();

 private:
  std::string name_;

  std::vector<nndeploy::device::Buffer *> buffers_;
  std::vector<nndeploy::device::Mat *> mats_;
  std::vector<nndeploy::device::Tensor *> tensors_;

  std::vector<nndeploy::base::Param *> params_;

#ifdef NNTASK_ENABLE_OPENCV
  std::vector<cv::mat *> cv_mats_;
#endif

  void *anything_ = nullptr;
};

}  // namespace common
}  // namespace nntask

#endif /* __NNTASK_SOURCE_COMMON_PACKET_H_ */
