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

struct DefaultBasePolicy {
  std::size_t operator()(std::size_t head, std::size_t min_tail,
                         bool has_consumers) const {
    return has_consumers ? min_tail : head;
  }
};

template <class Slot, class BasePolicy = DefaultBasePolicy>
class SpmcRingCore {
 public:
  SpmcRingCore(std::size_t cap, BasePolicy bp = {})
      : cap_(cap), mask_(cap - 1), buf_(cap), base_(bp) {
    assert(cap >= 2 && (cap_ & mask_) == 0);
  }
  SpmcRingCore(const SpmcRingCore &) = delete;
  SpmcRingCore &operator=(const SpmcRingCore &) = delete;

  bool push(const Slot &v) {
    return push_impl([&](std::optional<Slot> &s) { s.emplace(v); });
  }

  bool push(Slot &&v) {
    return push_impl([&](std::optional<Slot> &s) { s.emplace(std::move(v)); });
  }

  bool pop(std::size_t cid, Slot &out) {
    std::size_t t = tails_[cid].load(std::memory_order_relaxed);
    std::size_t h = head_.load(std::memory_order_acquire);
    if (h <= t) return false;
    out = *buf_[t & mask_];
    tails_[cid].store(t + 1, std::memory_order_release);
    return true;
  }

  std::size_t add_tails(std::size_t start_seq) {
    // tails_.push_back(std::atomic<std::size_t>(start_seq));
    tails_.emplace_back(start_seq);
    return tails_.size() - 1;
  }

  std::size_t head() const { return head_.load(std::memory_order_acquire); }
  std::size_t oldest() const {
    auto h = head();
    return (h > cap_) ? (h - cap_) : 0;
  }
  std::size_t cap() const { return cap_; }

 private:
  template <class F>
  bool push_impl(F &&write_into_optional) {
    std::size_t h = head_.load(std::memory_order_relaxed);
    auto [min_tail, has] = min_tail_relaxed_();
    std::size_t base = base_(h, min_tail, has);
    if (h - base >= cap_) return false;

    auto &opt = buf_[h & mask_];
    write_into_optional(opt);

    std::atomic_thread_fence(std::memory_order_release);
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  std::pair<std::size_t, bool> min_tail_relaxed_() const {
    if (tails_.empty()) return {0, false};
    std::size_t m = tails_[0].load(std::memory_order_relaxed);
    for (std::size_t i = 1; i < tails_.size(); ++i) {
      auto t = tails_[i].load(std::memory_order_relaxed);
      if (t < m) m = t;
    }
    return {m, true};
  }

 private:
  std::size_t cap_, mask_;
  std::vector<std::optional<Slot>> buf_;
  std::atomic<std::size_t> head_{0};
  std::deque<std::atomic<std::size_t>> tails_;

  BasePolicy base_;
};

struct FromOldest {
  static std::size_t start(std::size_t head, std::size_t oldest) {
    (void)head;
    return oldest;
  }
};

struct FromLatest {
  static std::size_t start(std::size_t head, std::size_t /*oldest*/) {
    return head;
  }
};

template <class Slot, class BasePolicy, class StartPolicy = FromOldest>
class SpmcRingQueue {
 public:
  SpmcRingQueue(std::size_t cap, BasePolicy bp = {})
      : core_(cap, std::move(bp)) {}

  std::size_t add() {
    return core_.add_tails(StartPolicy::start(core_.head(), core_.oldest()));
  }

  bool try_push(const Slot &v) {
    if (closed()) return false;
    bool ok = core_.push(v);
    if (ok) cv_not_empty_.notify_all();
    return ok;
  }
  bool try_push(Slot &&v) {
    if (closed()) return false;
    bool ok = core_.push(std::move(v));
    if (ok) cv_not_empty_.notify_all();
    return ok;
  }

  bool push(Slot v) {
    for (;;) {
      if (closed()) return false;
      if (core_.push(v)) {
        cv_not_empty_.notify_all();
        return true;
      }
      std::unique_lock<std::mutex> lk(mu_);
      // cv_not_full_.wait(lk, [&] { return closed() || likely_not_full_(); });
      cv_not_full_.wait(lk);
    }
  }

  bool try_pop(std::size_t cid, Slot &out) {
    bool ok = core_.pop(cid, out);
    if (ok) cv_not_full_.notify_all();
    return ok;
  }

  bool pop(std::size_t cid, Slot &out) {
    for (;;) {
      if (core_.pop(cid, out)) {
        cv_not_full_.notify_all();
        return true;
      }
      std::unique_lock<std::mutex> lk(mu_);
      if (closed()) {
        // 关闭后尝试最后再读一次；仍然没有就退出
        if (!core_.pop(cid, out)) return false;
        cv_not_full_.notify_all();
        return true;
      }
      // 被生产者写入或其他消费者提交时唤醒
      cv_not_empty_.wait(lk);
    }
  }

  void close() {
    closed_.store(true, std::memory_order_release);
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }
  bool closed() const { return closed_.load(std::memory_order_acquire); }
  size_t cap() const { return core_.cap(); }
  size_t head() const { return core_.head(); }
  size_t oldest() const { return core_.oldest(); }

 private:
  bool likely_not_full_() const {
    return (core_.head() - core_.oldest()) < core_.cap() || closed();
  }

 private:
  SpmcRingCore<Slot, BasePolicy> core_;
  std::atomic<bool> closed_{false};
  std::mutex mu_;
  std::condition_variable cv_not_full_, cv_not_empty_;
};

}  // namespace base
}  // namespace nndeploy

#endif