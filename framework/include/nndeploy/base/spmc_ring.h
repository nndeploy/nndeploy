#ifndef _NNDEPLOY_BASE_SPMC_RING_H_
#define _NNDEPLOY_BASE_SPMC_RING_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

template <typename T, typename Key = const void *>
class SpmcRing {
  static_assert(std::is_trivially_copyable<T>::value,
                "T should be trivially copyable or a pointer type.");

 public:
  struct ConsumerState {
    std::atomic<uint64_t> rseq{0};
    std::atomic<bool> consumed_once{false};
  };
  explicit SpmcRing(uint64_t capacity = 1) { set_capacity(capacity); }

  void set_capacity(uint64_t cap) {
    if (cap == 0) cap = 1;
    // 向上取 2^k，便于位运算取模
    uint64_t p2 = 1;
    while (p2 < cap) p2 <<= 1;
    capacity_ = p2;
    mask_ = capacity_ - 1;
    ring_.assign(capacity_, Slot{});
    write_seq_.store(0, std::memory_order_relaxed);
    published_seq_.store(0, std::memory_order_relaxed);
  }

  uint64_t capacity() const { return capacity_; }

  void register_consumer(Key k, bool start_from_latest = true) {
    uint64_t start = start_from_latest
                         ? published_seq_.load(std::memory_order_relaxed)
                         : write_seq_.load(std::memory_order_relaxed);
    auto &st = readers_[k];
    st.rseq.store(start, std::memory_order_relaxed);
    st.consumed_once.store(false, std::memory_order_relaxed);
  }

  void unregister_consumer(Key k) { readers_.erase(k); }

  bool try_push(const T &v) {
    uint64_t w = write_seq_.load(std::memory_order_relaxed);
    uint64_t min_r = min_reader_seq_acquire();
    if (w - min_r >= capacity_) return false;  // full

    ring_[w & mask_].payload = v;
    std::atomic_thread_fence(std::memory_order_release);
    published_seq_.store(w + 1, std::memory_order_release);
    write_seq_.store(w + 1, std::memory_order_relaxed);
    cv_not_empty_.notify_all();
    return true;
  }

  bool push_blocking(const T &v) {
    if (try_push(v)) return true;
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_full_.wait(lk, [&] {
      if (terminate_.load(std::memory_order_acquire)) return true;
      uint64_t w = write_seq_.load(std::memory_order_relaxed);
      uint64_t min_r = min_reader_seq_acquire();
      return (w - min_r) < capacity_;
    });
    if (terminate_.load(std::memory_order_acquire)) return false;
    return try_push(v);
  }

  bool has_new(Key k) const {
    auto it = readers_.find(k);
    if (it == readers_.end()) return false;
    uint64_t r = it->second.rseq.load(std::memory_order_relaxed);
    uint64_t p = published_seq_.load(std::memory_order_acquire);
    return p > r;
  }

  bool read_next(Key k, T &out, bool sticky = true,
                 bool blocking_first = true) {
    auto it = readers_.find(k);
    if (it == readers_.end()) {
      register_consumer(k, /*start_from_latest=*/true);
      it = readers_.find(k);
    }
    auto &st = it->second;

    uint64_t r = st.rseq.load(std::memory_order_relaxed);
    uint64_t p = published_seq_.load(std::memory_order_acquire);
    if (p > r) {
      out = ring_[r & mask_].payload;
      st.rseq.store(r + 1, std::memory_order_release);
      st.consumed_once.store(true, std::memory_order_release);
      cv_not_full_.notify_all();
      return true;
    }
    if (sticky && st.consumed_once.load(std::memory_order_acquire)) {
      uint64_t last = (r ? r - 1 : 0);
      out = ring_[last & mask_].payload;
      return false;  // 复用旧帧：返回 false 表示“非新”
    }
    if (!blocking_first) return false;

    // 等首帧
    std::unique_lock<std::mutex> lk(mtx_);
    cv_not_empty_.wait(lk, [&] {
      if (terminate_.load(std::memory_order_acquire)) return true;
      uint64_t rr = st.rseq.load(std::memory_order_relaxed);
      uint64_t pp = published_seq_.load(std::memory_order_acquire);
      return pp > rr;
    });
    if (terminate_.load(std::memory_order_acquire)) return false;

    r = st.rseq.load(std::memory_order_relaxed);
    out = ring_[r & mask_].payload;
    st.rseq.store(r + 1, std::memory_order_release);
    st.consumed_once.store(true, std::memory_order_release);
    cv_not_full_.notify_all();
    return true;
  }

  bool consumed_once(Key k) const {
    auto it = readers_.find(k);
    return it != readers_.end() &&
           it->second.consumed_once.load(std::memory_order_acquire);
  }

  void request_terminate() {
    terminate_.store(true, std::memory_order_release);
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }

  bool terminated() const { return terminate_.load(std::memory_order_acquire); }

 private:
  struct Slot {
    T payload{};
  };

  uint64_t min_reader_seq_acquire() const {
    uint64_t m = std::numeric_limits<uint64_t>::max();
    for (auto &kv : readers_) {
      uint64_t r = kv.second.rseq.load(std::memory_order_acquire);
      if (r < m) m = r;
    }
    return (readers_.empty() ? write_seq_.load(std::memory_order_relaxed) : m);
  }

 private:
  // ring
  std::vector<Slot> ring_;
  uint64_t capacity_{1};
  uint64_t mask_{0};

  // seq
  std::atomic<uint64_t> write_seq_{0};
  std::atomic<uint64_t> published_seq_{0};

  // readers
  std::unordered_map<Key, ConsumerState> readers_;

  // waits
  mutable std::mutex mtx_;
  std::condition_variable cv_not_full_, cv_not_empty_;
  std::atomic<bool> terminate_{false};
};

}  // namespace base
}  // namespace nndeploy

#endif