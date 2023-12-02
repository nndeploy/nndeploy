
#ifndef _NNDEPLOY_THREAD_POOL_SAFE_WS_QUEUE_H_
#define _NNDEPLOY_THREAD_POOL_SAFE_WS_QUEUE_H_

#include <deque>
#include <thread>

namespace nndeploy {
namespace thread_pool {

template <typename T>
class SafeWSQueue {
 public:
  /**
   * 尝试往队列里写入信息
   * @param task
   * @return
   */
  bool tryPush(T&& task) {
    bool result = false;
    if (lock_.try_lock()) {
      deque_.emplace_back(std::forward<T>(task));
      lock_.unlock();
      result = true;
    }
    return result;
  }

  /**
   * 弹出节点，从头部进行
   * @param task
   * @return
   */
  bool tryPop(T& task) {
    bool result = false;
    if (!deque_.empty() && lock_.try_lock()) {
      if (!deque_.empty()) {
        task = std::forward<T>(deque_.front());  // 从前方弹出
        deque_.pop_front();
        result = true;
      }
      lock_.unlock();
    }

    return result;
  }

  /**
   * 窃取节点，从尾部进行
   * @param task
   * @return
   */
  bool trySteal(T& task) {
    bool result = false;
    if (!deque_.empty() && lock_.try_lock()) {
      if (!deque_.empty()) {
        task = std::forward<T>(deque_.back());  // 从后方窃取
        deque_.pop_back();
        result = true;
      }
      lock_.unlock();
    }

    return result;
  }

  SafeWSQueue() = default;

 private:
  std::deque<T> deque_;  // 存放任务的双向队列
  std::mutex lock_;
};

}  // namespace thread_pool
}  // namespace nndeploy

#endif  //_NNDEPLOY_THREAD_POOL_SAFE_WS_QUEUE_H_
