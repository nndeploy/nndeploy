
#include "nndeploy/thread_pool/parallel_for_api_default.h"

#include "nndeploy/thread_pool/parallel_for_api.h"

namespace nndeploy {
namespace thread_pool {

class ParallelJob {
 public:
  ParallelJob(const ParallelPool& parallel_pool, const base::Range& range,
              const ParallelLoopBody& body, int nstripes)
      : cur_position_(0),
        active_thread_num_(0),
        completed_thread_num_(0),
        completed_(false),
        parallel_pool_(parallel_pool),
        range_(range),
        body_(body),
        nstripes_(nstripes) {}
  ~ParallelJob() {}

  int run() {
    int total_cnt = range_.size();
    int step = 1;

    if (nstripes_ <= 1) {
      step = std::max(
          (total_cnt - cur_position_) / (parallel_pool_.getThreadNum() * 2), 1);
    } else {
      step = nstripes_;
    }

    for (;;) {
      int start = cur_position_.fetch_add(step, std::memory_order_seq_cst);
      if (start >= total_cnt) {
        break;
      }

      int end = std::min(total_cnt, start + step);

      body_(base::Range(range_.start_ + start, range_.start_ + end));
    }

    return 0;
  }

  const base::Range& range() { return range_; }

 public:
  std::atomic<int> cur_position_;
  int64_t placeholder1_[8];

  std::atomic<int> active_thread_num_;
  int64_t placeholder2_[8];

  std::atomic<int> completed_thread_num_;
  int64_t placeholder3_[8];

  std::atomic<bool> completed_;

 private:
  const ParallelPool& parallel_pool_;
  const base::Range range_;
  const ParallelLoopBody& body_;
  const int nstripes_;
};

class WorkerThread {
 public:
  WorkerThread(ParallelPool& parallel_pool, unsigned int thread_id)
      : awake_(false),
        is_active_(true),
        parallel_pool_(parallel_pool),
        thread_id_(thread_id),
        stop_(false),
        active_wait_(2000) {
    thread_ = std::thread(
        [](void* work_thread) { ((WorkerThread*)work_thread)->loop_body(); },
        this);
  }

  ~WorkerThread() {
    if (!stop_) {
      std::unique_lock<std::mutex> lock(thread_mutex_);
      stop_ = true;
      lock.unlock();
    }

    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void stop() { stop_ = true; }

  void loop_body() {
    bool allow_active_wait = true;
    while (!stop_) {
      // if (thread_ > 0)
      {
        allow_active_wait = false;
        for (int i = 0; i < active_wait_; i++) {
          if (awake_) {
            break;
          }

          std::this_thread::yield();
        }
      }

      std::unique_lock<std::mutex> lock(thread_mutex_);
      while (!awake_) {
        is_active_ = false;
        cond_thread_wake_.wait(lock);
        is_active_ = true;
      }

      allow_active_wait = true;

      std::shared_ptr<ParallelJob> j_ptr;
      swap(j_ptr, job_);
      awake_ = false;
      lock.unlock();

      if (stop_ || !j_ptr || j_ptr->cur_position_ >= j_ptr->range().size()) {
        continue;
      }

      j_ptr->active_thread_num_.fetch_add(1, std::memory_order_seq_cst);
      j_ptr->run();

      int completed_ =
          j_ptr->completed_thread_num_.fetch_add(1, std::memory_order_seq_cst) +
          1;
      int active = j_ptr->active_thread_num_.load(std::memory_order_acquire);

      if (active == completed_) {
        bool need_notify = !j_ptr->completed_;
        j_ptr->completed_ = true;
        j_ptr = nullptr;
        if (need_notify) {
          std::unique_lock<std::mutex> tlock(parallel_pool_.mutex_notify_);
          tlock.unlock();
          parallel_pool_.job_complete_.notify_all();
        }
      }
    }
  }

 public:
  std::atomic<bool> awake_;
  std::shared_ptr<ParallelJob> job_;
  std::condition_variable cond_thread_wake_;
  volatile bool is_active_;
  std::mutex thread_mutex_;

 private:
  ParallelPool& parallel_pool_;
  const unsigned int thread_id_;
  std::thread thread_;
  std::atomic<bool> stop_;
  const int active_wait_;
};

ParallelPool::ParallelPool() : active_wait_(10000) { thread_num_ = 2; }

ParallelPool::~ParallelPool() { setWorkThreads(0); }

void ParallelPool::setThreadNum(int num) {
  if (num == thread_num_) {
    return;
  }

  std::unique_lock<std::mutex> lock(pool_mutex_);
  thread_num_ = std::max(num, 1);
  setWorkThreads(thread_num_ - 1);
}

int ParallelPool::getThreadNum() const { return thread_num_; }

void ParallelPool::parallelFor(const base::Range& range,
                               const ParallelLoopBody& body, double nstripes) {
  std::unique_lock<std::mutex> lock(pool_mutex_);

  if (thread_num_ <= 1 || job_ || nstripes == 1) {
    lock.unlock();
    body(range);
    return;
  }

  setWorkThreads(thread_num_ - 1);
  job_ = std::shared_ptr<ParallelJob>(
      new ParallelJob(*this, range, body, nstripes));
  lock.unlock();

  for (size_t i = 0; i < work_threads_.size(); ++i) {
    WorkerThread& thread = *(work_threads_[i].get());
    if (thread.is_active_ || thread.awake_ || thread.job_) {
      std::unique_lock<std::mutex> lock(thread.thread_mutex_);
      thread.job_ = job_;
      bool is_active_ = thread.is_active_;
      thread.awake_ = true;
      lock.unlock();
      if (!is_active_) {
        thread.cond_thread_wake_.notify_all();
      }
    } else {
      thread.job_ = job_;
      thread.awake_ = true;
      thread.cond_thread_wake_.notify_all();
    }
  }

  job_->run();

  if (job_->completed_ || job_->active_thread_num_ == 0) {
    job_->completed_ = true;
  } else {
    if (active_wait_ > 0) {
      for (int i = 0; i < active_wait_; i++) {
        if (job_->completed_) {
          break;
        }

        std::this_thread::yield();
      }
    }

    if (!job_->completed_) {
      std::unique_lock<std::mutex> tlock(mutex_notify_);
      for (;;) {
        if (job_->completed_) {
          break;
        }
        job_complete_.wait(tlock);
      }
    }
  }

  if (job_) {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    job_ = nullptr;
  }
}

bool ParallelPool::setWorkThreads(int num) {
  if (num == work_threads_.size()) {
    return false;
  }

  if (num < work_threads_.size()) {
    std::vector<std::shared_ptr<WorkerThread> > release_threads(
        work_threads_.size() - num);
    for (size_t i = num; i < work_threads_.size(); ++i) {
      std::unique_lock<std::mutex> lock(work_threads_[i]->thread_mutex_);
      work_threads_[i]->stop();
      work_threads_[i]->awake_ = true;
      lock.unlock();
      work_threads_[i]->cond_thread_wake_.notify_all();
      std::swap(work_threads_[i], release_threads[i - num]);
    }
    work_threads_.resize(num);
    release_threads.clear();
    return false;
  } else {
    for (size_t i = work_threads_.size(); i < num; ++i) {
      work_threads_.push_back(
          std::shared_ptr<WorkerThread>(new WorkerThread(*this, (unsigned)i)));
    }
  }
  return false;
}

void ParallelForApiDefault::setThreadNum(int num) {
  ParallelPool::getInstance().setThreadNum(num);
}

int ParallelForApiDefault::getThreadNum() {
  return ParallelPool::getInstance().getThreadNum();
}

int ParallelForApiDefault::parallelFor(const base::Range& range,
                                       const ParallelLoopBody& body,
                                       double nstripes) {
  ParallelPool::getInstance().parallelFor(range, body, nstripes);
  return 0;
}

}  // namespace thread_pool
}  // namespace nndeploy