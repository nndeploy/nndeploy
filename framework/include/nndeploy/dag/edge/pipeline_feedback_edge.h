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

  bool hasBeenConsumedBy(const Node *n) override { return false; }

  bool empty() override;

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

  std::mutex mutex_;
  std::condition_variable cv_;

  std::map<Node *, int> to_consume_index_;
};

}  // namespace dag
}  // namespace nndeploy

#endif