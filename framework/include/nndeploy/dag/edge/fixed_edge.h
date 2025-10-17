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
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class FixedEdge : public AbstractEdge {
 public:
  FixedEdge(base::ParallelType paralle_type);
  virtual ~FixedEdge();

  virtual base::Status setQueueMaxSize(int queue_max_size);

  virtual bool empty();

  virtual base::Status construct();

  virtual base::Status set(device::Buffer *buffer, bool is_external);
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc);
  virtual bool notifyWritten(device::Buffer *buffer);
  virtual device::Buffer *getBuffer(const Node *node);
  virtual device::Buffer *getGraphOutputBuffer();

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, bool is_external);
  virtual cv::Mat *create(int rows, int cols, int type, const cv::Vec3b &value);
  virtual bool notifyWritten(cv::Mat *cv_mat);
  virtual cv::Mat *getCvMat(const Node *node);
  virtual cv::Mat *getGraphOutputCvMat();
#endif

  virtual base::Status set(device::Tensor *tensor, bool is_external);
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc,
                                 const std::string &name);
  virtual bool notifyWritten(device::Tensor *tensor);
  virtual device::Tensor *getTensor(const Node *node);
  virtual device::Tensor *getGraphOutputTensor();

  virtual base::Status takeDataPacket(DataPacket *data_packet);
  virtual bool notifyWritten(void *anything);
  virtual DataPacket *getDataPacket(const Node *node);
  virtual DataPacket *getGraphOutputDataPacket();

  virtual base::Status set(base::Param *param, bool is_external);
  virtual bool notifyWritten(base::Param *param);
  virtual base::Param *getParam(const Node *node);
  virtual base::Param *getGraphOutputParam();

  virtual int64_t getIndex(const Node *node);
  virtual int64_t getGraphOutputIndex();

  virtual int getPosition(const Node *node);
  virtual int getGraphOutputPosition();

  virtual base::EdgeUpdateFlag update(const Node *node);

  virtual bool requestTerminate();

 private:
  DataPacket *data_packet_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
