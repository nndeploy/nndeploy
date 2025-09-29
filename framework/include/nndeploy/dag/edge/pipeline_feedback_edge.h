#ifndef _NNDEPLOY_DAG_EDGE_PIPELINE_FEEDBACK_EDGE_H_
#define _NNDEPLOY_DAG_EDGE_PIPELINE_FEEDBACK_EDGE_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/spmc_ring.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

static inline std::size_t round_up_pow2(std::size_t x) {
  if (x < 2) return 2;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
#if INTPTR_MAX == INT64_MAX
  x |= x >> 32;
#endif
  return x + 1;
}

class PipelineFeedbackEdge : public AbstractEdge {
 public:
  PipelineFeedbackEdge(base::ParallelType parallel_type);
  ~PipelineFeedbackEdge() override;

  base::Status construct() override;
  base::Status setQueueMaxSize(int q) override;

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

  bool hasBeenConsumedBy(const Node *n) override { return false; }

 private:
  using Slot = std::shared_ptr<DataPacket>;
  using Ring = nndeploy::base::SpmcRingQueue<Slot, const Node *,
                                             nndeploy::base::DefaultBasePolicy>;

  // ring queue
  std::unique_ptr<Ring> ring_;
  // max queue size
  std::size_t queue_max_size_{1024};
  // consumer number
  int consumer_size_;
};

}  // namespace dag
}  // namespace nndeploy

#endif