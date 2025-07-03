#ifndef _NNDEPLOY_DAG_EDGE_PIPELINE_EDGE_RB_H_
#define _NNDEPLOY_DAG_EDGE_PIPELINE_EDGE_RB_H_

#include <atomic>
#include <cassert>
#include <cstdint>
#include <deque>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge/abstract_edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

template <typename Slot>
class RingBufferSpmc {
 public:
  explicit RingBufferSpmc(std::size_t cap)
      : cap_(cap), mask_(cap - 1), buf_(cap) {
    assert(cap >= 2 && (cap & mask_) == 0);
  }

  Slot* reserve() {
    std::size_t h = head_.load(std::memory_order_relaxed);
    std::size_t min_tail = min_tail_cached();

    if (h - min_tail >= cap_) return nullptr;
    return &buf_[h & mask_];
  }
  void commit() { head_.fetch_add(1, std::memory_order_release); }

  Slot* peek(std::size_t cid) {
    std::size_t t = tails_[cid].load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return nullptr;
    return &buf_[t & mask_];
  }
  void pop(std::size_t cid) {
    tails_[cid].fetch_add(1, std::memory_order_release);
  }

  std::size_t register_consumer() {
    tails_.push_back(std::atomic<std::size_t>(0));
    return tails.size() - 1;
  }
  std::size_t size() const {
    return head_.load(std::memory_order_acquire) - min_tail_cached();
  }

 private:
  std::size_t min_tail_cached() const {
    std::size_t min_t = tails_[0].load(std::memory_order_relaxed);
    for (std::size_t i = 1; i < tails_.size(); ++i) {
      std::size_t t = tails_[i].load(std::memory_order_relaxed);
      if (t < min_t) min_t = t;
    }
    return min_t;
  }

  const std::size_t cap_, mask_;
  std::vector<Slot> buf_;
  std::atomic<std::size_t> head_{0};
  std::deque<std::atomic<std::size_t>> tails_;
};

// class PipelineEdge : public AbstractEdge {
//  public:
//   PipelineEdge(base::ParallelType paralle_type);
//   virtual ~PipelineEdge();

//   virtual base::Status setQueueMaxSize(int q) override;
//   virtual base::Status construct() override;

//   // buffer
//   virtual base::Status set(device::Buffer* buf, bool ext) override;
//   virtual device::Buffer* create(device::Device* dev,
//                                  const device::BufferDesc& desc) override;
//   virtual bool notifyWritten(device::Buffer* buf) override;
//   virtual device::Buffer* getBuffer(const Node* n) override;
//   virtual device::Buffer* getGraphOutputBuffer() override;

//   // cv_mat
// #ifdef ENABLE_NNDEPLOY_OPENCV
//   virtual base::Status set(cv::Mat* cv_mat, bool ext) override;
//   virtual cv::Mat* create(int rows, int cols, int type,
//                           const cv::Vec3b& value) override;
//   virtual bool notifyWritten(cv::Mat* cv_mat) override;
//   virtual device::Buffer* getCvMat(const Node* n) override;
//   virtual device::Buffer* getGraphOutputCvMat() override;
// #endif

//   // tensor
//   virtual base::Status set(device::Tensor* tensor, bool ext) override;
//   virtual device::Tensor* create(device::Device* dev,
//                                  const device::TensorDesc& desc,
//                                  const std::string& name) override;
//   virtual bool notifyWritten(device::Tensor* tensor) override;
//   virtual device::Tensor* getTensor(const Node* n) override;
//   virtual device::Tensor* getGraphOutputTensor() override;

//   // param
//   virtual base::Status set(base::Param* param, bool is_external) override;
//   virtual bool notifyWritten(base::Param* param) override;
//   virtual base::Param* getParam(const Node* node) override;
//   virtual base::Param* getGraphOutputParam() override;

//   // data packet
//   virtual base::Status takeDataPacket(DataPacket* packet) override;
//   virtual bool notifyAnyWritten(void* anything) override;
//   virtual DataPacket* getDataPacket(const Node* node) override;
//   virtual DataPacket* getGraphOutputDataPacket() override;

//   virtual int64_t getIndex(const Node* node);
//   virtual int64_t getGraphOutputIndex();

//   virtual int getPosition(const Node* node);
//   virtual int getGraphOutputPosition();

//   virtual base::EdgeUpdateFlag update(const Node* node);
//   virtual bool requestTerminate();

//  private:
//   PipelineDataPacket* getPipelineDataPacket(const Node* node);

//   RingBufferSpmc<PipelineDataPacket> rb_;
//   std::unordered_map<const Node*, std::size_t> consumer_id_;
//   std::unordered_map<const Node*, PipelineDataPacket*> consuming_dp_;
//   int consumers_size_ = 0;
//   int queue_max_size_ = 128;
//   std::atomic<bool> terminate_{false};
// };

}  // namespace dag
}  // namespace nndeploy

#endif