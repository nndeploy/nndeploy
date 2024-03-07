#ifndef _NNDEPLOY_DAG_EDGE_FIXED_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_FIXED_EDGE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/dag/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class FixedEdge : public AbstractEdge {
 public:
  FixedEdge(ParallelType paralle_type);
  virtual ~FixedEdge();

  virtual base::Status construct();

  virtual base::Status set(device::Buffer *buffer, int index, bool is_external);
  virtual base::Status set(device::Buffer &buffer, int index);
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc, int index);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer(const Node *node);
  virtual device::Buffer *getGraphOutputBuffer();

  virtual base::Status set(device::Mat *mat, int index, bool is_external);
  virtual base::Status set(device::Mat &mat, int index);
  virtual device::Mat *create(device::Device *device,
                              const device::MatDesc &desc, int index,
                              const std::string &name);
  virtual bool notifyWritten(device::Mat *mat);
  virtual device::Mat *getMat(const Node *node);
  virtual device::Mat *getGraphOutputMat();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, int index, bool is_external);
  virtual base::Status set(cv::Mat &cv_mat, int index);
  virtual cv::Mat *getCvMat(const Node *node);
  virtual cv::Mat *getGraphOutputCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, int index, bool is_external);
  virtual base::Status set(device::Tensor &tensor, int index);
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc, int index,
                                 const std::string &name);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor(const Node *node);
  virtual device::Tensor *getGraphOutputTensor();

  virtual base::Status set(base::Param *param, int index, bool is_external);
  virtual base::Status set(base::Param &param, int index);
  virtual base::Param *getParam(const Node *node);
  virtual base::Param *getGraphOutputParam();

  virtual base::Status set(void *anything, int index, bool is_external);
  virtual void *getAnything(const Node *node);
  virtual void *getGraphOutputAnything();

  virtual int getIndex(const Node *node);
  virtual int getGraphOutputIndex();

  virtual int getPosition(const Node *node);
  virtual int getGraphOutputPosition();

  virtual EdgeUpdateFlag update(const Node *node);

  virtual bool requestTerminate();

 private:
  DataPacket *data_packet_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
