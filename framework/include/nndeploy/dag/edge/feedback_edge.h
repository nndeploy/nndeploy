#ifndef _NNDEPLOY_DAG_EDGE_FEEDBACK_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_FEEDBACK_EDGE_H_

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

class FeedBackEdge : public AbstractEdge {
 public:
  FeedBackEdge(base::ParallelType paralle_type);
  ~FeedBackEdge() override;

  base::Status construct() override;

  base::Status set(device::Buffer *buffer, bool is_external) override;
  device::Buffer *create(device::Device *device,
                         const device::BufferDesc &desc) override;
  bool notifyWritten(device::Buffer *buffer) override;
  device::Buffer *getBuffer(const Node *node) override;
  device::Buffer *getGraphOutputBuffer() override;

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat *cv_mat, bool is_external) override;
  cv::Mat *create(int rows, int cols, int type,
                  const cv::Vec3b &value) override;
  bool notifyWritten(cv::Mat *cv_mat) override;
  cv::Mat *getCvMat(const Node *node) override;
  cv::Mat *getGraphOutputCvMat() override;
#endif

  base::Status set(device::Tensor *tensor, bool is_external) override;
  device::Tensor *create(device::Device *device, const device::TensorDesc &desc,
                         const std::string &name) override;
  bool notifyWritten(device::Tensor *tensor) override;
  device::Tensor *getTensor(const Node *node) override;
  device::Tensor *getGraphOutputTensor() override;

  base::Status takeDataPacket(DataPacket *data_packet) override;
  bool notifyWritten(void *anything) override;
  DataPacket *getDataPacket(const Node *node) override;
  DataPacket *getGraphOutputDataPacket() override;

  base::Status set(base::Param *param, bool is_external) override;
  bool notifyWritten(base::Param *param) override;
  base::Param *getParam(const Node *node) override;
  base::Param *getGraphOutputParam() override;

  int64_t getIndex(const Node *node) override;
  int64_t getGraphOutputIndex() override;

  int getPosition(const Node *node) override;
  int getGraphOutputPosition() override;

  base::EdgeUpdateFlag update(const Node *node) override;

  bool requestTerminate() override;

  base::Status setQueueMaxSize(int queue_max_size) override;

  bool hasBeenConsumedBy(const Node *n) override;

  bool empty() override;

 private:
  DataPacket *data_packet_;
  std::unordered_map<const Node *, int64_t> last_read_index_;
};

}  // namespace dag
}  // namespace nndeploy

#endif
