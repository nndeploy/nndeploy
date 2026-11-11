
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

  virtual base::Status setQueueMaxSize(int queue_max_size) override;

  virtual bool empty() override;

  virtual base::Status construct() override;

  virtual base::Status set(device::Buffer *buffer, bool is_external) override;
  virtual device::Buffer *create(device::Device *device,
                                 const device::BufferDesc &desc) override;
  virtual bool notifyWritten(device::Buffer *buffer) override;
  virtual device::Buffer *getBuffer(const Node *node) override;
  virtual device::Buffer *getGraphOutputBuffer() override;

#ifdef ENABLE_NNDEPLOY_OPENCV
  virtual base::Status set(cv::Mat *cv_mat, bool is_external) override;
  virtual cv::Mat *create(int rows, int cols, int type, const cv::Vec3b &value) override;
  virtual bool notifyWritten(cv::Mat *cv_mat) override;
  virtual cv::Mat *getCvMat(const Node *node) override;
  virtual cv::Mat *getGraphOutputCvMat() override;
#endif

  virtual base::Status set(device::Tensor *tensor, bool is_external) override;
  virtual device::Tensor *create(device::Device *device,
                                 const device::TensorDesc &desc,
                                 const std::string &name) override;
  virtual bool notifyWritten(device::Tensor *tensor) override;
  virtual device::Tensor *getTensor(const Node *node) override;
  virtual device::Tensor *getGraphOutputTensor() override;

  virtual base::Status takeDataPacket(DataPacket *data_packet) override;
  virtual bool notifyWritten(void *anything) override;
  std::shared_ptr<DataPacket> acquirePacket(const Node *node, bool graph_output,
      EdgeTypeFlag expected = EdgeTypeFlag::kNone, const std::type_info *type = nullptr) override;
  virtual DataPacket *getDataPacket(const Node *node) override;
  virtual DataPacket *getGraphOutputDataPacket() override;

  virtual base::Status set(base::Param *param, bool is_external) override;
  virtual bool notifyWritten(base::Param *param) override;
  virtual base::Param *getParam(const Node *node) override;
  virtual base::Param *getGraphOutputParam() override;

  virtual void *getGraphOutputPtr() override;

  virtual int64_t getIndex(const Node *node) override;
  virtual int64_t getGraphOutputIndex() override;

  virtual int getPosition(const Node *node) override;
  virtual int getGraphOutputPosition() override;

  virtual base::EdgeUpdateFlag update(const Node *node) override;

  virtual bool requestTerminate() override;

 private:
  std::mutex mutex_;
  std::shared_ptr<DataPacket> data_packet_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
