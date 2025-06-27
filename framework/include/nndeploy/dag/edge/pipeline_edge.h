#ifndef _NNDEPLOY_DAG_EDGE_PIPELINE_EDGE_RB_H_
#define _NNDEPLOY_DAG_EDGE_PIPELINE_EDGE_RB_H_

#include <atomic>
#include <cassert>
#include <cstdint>
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
  std::vector<std::atomic<std::size_t>> tails_;
};

enum class SlotState : uint8_t { kEmpty = 0, kWriting = 1, kReady = 2 };

struct DataSlot {
  std::atomic<SlotState> state{SlotState::kEmpty};
  std::shared_ptr<PipelineDataPacket> pkt;
};

class PipelineEdge final : public AbstractEdge {
 public:
  explicit PipelineEdge(base::ParallelType pt)
      : AbstractEdge(pt),
        ring_(std::make_unique<RingBufferSpmc<DataSlot>>(1024)) {}

  ~PipelineEdge() override {
    terminate_flag_.store(true, std::memory_order_relaxed);
  }

  base::Status setQueueMaxSize(int q) override {
    queue_max_size_ = q;
    ring_ = std::make_unique<RingBufferSpmc<DataSlot>>(round_up_pow2(q));
    return base::kStatusCodeOk;
  }

  base::Status construct() override {
    consumers_size_ = static_cast<int>(consumers_.size());
    for (auto* n : consumers_) {
      consumer_id_[n] = ring_->register_consumer();
    }
    return base::kStatusCodeOk;
  }

  base::Status set(device::Buffer* buf, bool ext) override {
    auto pkt = std::make_shared<PipelineDataPacket>(consumers_size_);
    NNDEPLOY_RETURN_ON_NEQ(pkt->set(buf, ext), base::kStatusCodeOk, "pkt::set");
    return push_ready(std::move(pkt));
  }

  device::Buffer* create(device::Device* dev,
                         const device::BufferDesc& desc) override {
    auto pkt = reserve_writing();
    if (!pkt) return nullptr;
    auto* out = pkt->create(dev, desc);
    if (out) buffer2pkt_[out] = pkt;
    return out;
  }

  bool notifyWritten(device::Buffer* buf) override {
    return commit_written(buf);
  }

  device::Buffer* getBuffer(const Node* n) override {
    return current_pkt_[n] ? current_pkt_[n]->getBuffer() : nullptr;
  }
  device::Buffer* getGraphOutputBuffer() override { return getBuffer(nullptr); }

#ifdef ENABLE_NNDEPLOY_OPENCV
  base::Status set(cv::Mat* cv_mat, bool ext) override {
    auto pkt = std::make_shared<PipelineDataPacket>(consumers_size_);
    NNDEPLOY_RETURN_ON_NEQ(pkt->set(cv_mat, ext), base::kStatusCodeOk,
                           "pkt::set");
    return push_ready(std::move(pkt));
  }

  cv::Mat* create(int rows, int cols, int type, const cv::Vec3b& value) {
    auto pkt = reserve_writing();
    if (!pkt) return nullptr;
    auto* out = pkt->create(rows, cols, type, value);
    if (out) buffer2pkt_[out] = pkt;
    return out;
  }

  bool notifyWritten(cv::Mat* cv_mat) override {
    return commit_written(cv_mat);
  }

  device::Buffer* getCvMat(const Node* n) override {
    return current_pkt_[n] ? current_pkt_[n]->getCvMat() : nullptr;
  }
  device::Buffer* getGraphOutputBuffer() override { return getCvMat(nullptr); }
#endif

  base::EdgeUpdateFlag update(const Node* node) override {
    Node* n = const_cast<Node*>(node);
    if (!checkNode(n)) return base::kEdgeUpdateFlagError;

    const std::size_t cid = consumer_id_[n];
    while (true) {
      if (terminate_flag_.load(std::memory_order_acquire))
        return base::kEdgeUpdateFlagTerminate;

      DataSlot* s = ring_->peek(cid);
      if (!s) {
        std::this_thread::yield();
        continue;
      }

      if (s->state.load(std::memory_order_acquire) != SlotState::kReady) {
        std::this_thread::yield();
        continue;
      }
      current_pkt_[n] = s->pkt.get();  // 保存裸指针供 get*
      ring_->pop(cid);                 // tail++
      /* 回收：最后一个消费者离开时复位槽 */
      if (s->pkt.use_count() == 1) {  // 仅 pipeline 还持有
        s->pkt.reset();
        s->state.store(SlotState::kEmpty, std::memory_order_relaxed);
      }
      return base::kEdgeUpdateFlagComplete;
    }
  }

  bool requestTerminate() override {
    terminate_flag_.store(true, std::memory_order_release);
    return true;
  }

 private:
  base::Status push_ready(std::shared_ptr<PipelineDataPacket>&& pkt) {
    DataSlot* s = wait_reserve();
    s->pkt = std::move(pkt);
    s->state.store(SlotState::kReady, std::memory_order_release);
    ring_->commit();
    return base::kStatusCodeOk;
  }

  std::shared_ptr<PipelineDataPacket> reserve_writing() {
    DataSlot* s = wait_reserve();
    s->state.store(SlotState::kWriting, std::memory_order_relaxed);
    s->pkt.reset(new PipelineDataPacket(consumers_size_));
    return s->pkt.get();
  }

  bool commit_written(device::Buffer* buf) {
    auto it = buffer2pkt_.find(buf);
    if (it == buffer2pkt_.end()) return false;
    auto pkt = it->second.lock();
    if (!pkt) return false;
    pkt->notifyWritten(buf);  // 调原逻辑
    DataSlot* s = pkt2slot(pkt.get());
    if (!s) return false;
    s->state.store(SlotState::kReady, std::memory_order_release);
    buffer2pkt_.erase(it);
    return true;
  }

  DataSlot* wait_reserve() {
    while (true) {
      if (auto* s = ring_->reserve()) return s;
      std::this_thread::yield();  // 满–>背压
    }
  }
  DataSlot* pkt2slot(PipelineDataPacket* p) {
    // O(cap) 线性扫描；可换成更快 map
    for (auto& s : ring_->buf_)
      if (s.pkt.get() == p) return &s;
    return nullptr;
  }

  static std::size_t round_up_pow2(std::size_t v) {
    if (v < 2) return 2;
    --v;
    for (int i = 1; i < sizeof(std::size_t) * 8; i <<= 1) v |= v >> i;
    return ++v;
  }

  std::unique_ptr<RingBufferSpmc<DataSlot>> ring_;
  std::unordered_map<const Node*, std::size_t> consumer_id_;
  std::unordered_map<const Node*, PipelineDataPacket*> current_pkt_;
  std::unordered_map<void*, std::weak_ptr<PipelineDataPacket>> buffer2pkt_;
  std::atomic<bool> terminate_flag_{false};
  int consumers_size_{0};
  int queue_max_size_{1024};
};

}  // namespace dag
}  // namespace nndeploy

#endif