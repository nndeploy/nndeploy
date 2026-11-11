
#include "nndeploy/thread_pool/parallel_for_api_default.h"

#include "nndeploy/thread_pool/parallel_for_api.h"

namespace nndeploy {
namespace thread_pool {

class ParallelJob {
 public:
  ParallelJob(const base::Range &range, const ParallelLoopBody &body,
              int stripes, int threads)
      : range_(range), body_(body), stripes_(stripes), threads_(threads),
        pending_(threads - 1) {}

  void run() {
    try {
      int total = range_.size();
      int step = stripes_ > 1 ? stripes_ :
          std::max<int64_t>((total - next_.load()) / (2LL * threads_), 1);
      for (auto start = next_.fetch_add(step); start < total;
           start = next_.fetch_add(step)) {
        body_(base::Range(range_.start_ + start,
                          range_.start_ + std::min<int64_t>(total, start + step)));
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!error_) error_ = std::current_exception();
    }
  }

  void workerDone() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (--pending_ == 0) done_.notify_one();
  }

  std::exception_ptr wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    return error_;
  }

 private:
  base::Range range_;
  const ParallelLoopBody &body_;
  int stripes_, threads_;
  std::atomic<int64_t> next_{0};
  int pending_;
  std::mutex mutex_;
  std::condition_variable done_;
  std::exception_ptr error_;
};

class WorkerThread {
 public:
  WorkerThread() : thread_([this] { loop(); }) {}
  ~WorkerThread() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_one();
    thread_.join();
  }

  void submit(const std::shared_ptr<ParallelJob> &job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = job;
    }
    wake_.notify_one();
  }

 private:
  void loop() {
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || job_; });
      if (stopping_) return;
      auto job = std::move(job_);
      lock.unlock();
      job->run();
      job->workerDone();
    }
  }
  std::mutex mutex_;
  std::condition_variable wake_;
  std::shared_ptr<ParallelJob> job_;
  bool stopping_ = false;
  std::thread thread_;
};

ParallelPool::ParallelPool() : thread_num_(2) {}
ParallelPool::~ParallelPool() = default;

void ParallelPool::stop() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  stop_requested_ = true;
  if (!job_) setWorkThreads(0);
}

void ParallelPool::setThreadNum(int num) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  stop_requested_ = false;
  thread_num_ = std::max(num, 1);
  if (!job_) setWorkThreads(thread_num_ - 1);
}

int ParallelPool::getThreadNum() const {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  return thread_num_;
}

void ParallelPool::parallelFor(const base::Range &range,
                               const ParallelLoopBody &body, double nstripes) {
  std::unique_lock<std::mutex> lock(pool_mutex_);
  if (thread_num_ <= 1 || job_ || nstripes == 1) {
    lock.unlock();
    body(range);
    return;
  }

  setWorkThreads(thread_num_ - 1);
  stop_requested_ = false;
  auto job = std::make_shared<ParallelJob>(range, body, nstripes, thread_num_);
  job_ = job;
  for (auto &worker : work_threads_) worker->submit(job);
  lock.unlock();
  job->run();
  auto error = job->wait();
  lock.lock();
  job_.reset();
  setWorkThreads(stop_requested_ ? 0 : thread_num_ - 1);
  if (error) std::rethrow_exception(error);
}

void ParallelPool::setWorkThreads(int num) {
  while (work_threads_.size() > num) work_threads_.pop_back();
  while (work_threads_.size() < num)
    work_threads_.emplace_back(std::make_shared<WorkerThread>());
}

void ParallelForApiDefault::setThreadNum(int num) {
  ParallelPool::getInstance().setThreadNum(num);
}

int ParallelForApiDefault::getThreadNum() {
  return ParallelPool::getInstance().getThreadNum();
}

int ParallelForApiDefault::parallelFor(const base::Range &range,
                                       const ParallelLoopBody &body,
                                       double nstripes) {
  ParallelPool::getInstance().parallelFor(range, body, nstripes);
  return 0;
}

}  // namespace thread_pool
}  // namespace nndeploy
