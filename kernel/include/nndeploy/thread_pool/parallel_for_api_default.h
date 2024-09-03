
#ifndef _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_DEFAULT_H_
#define _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_DEFAULT_H_

#include "nndeploy/thread_pool/parallel.h"
#include "nndeploy/thread_pool/parallel_for_api.h"

namespace nndeploy {
namespace thread_pool {

class ParallelJob;
class WorkerThread;

class ParallelPool {
 public:
  static ParallelPool &getInstance() {
    static ParallelPool *instance = new ParallelPool;
    return *instance;
  }

  ParallelPool();

  ~ParallelPool();

  void setThreadNum(int num);

  int getThreadNum() const;

  void parallelFor(const base::Range &range, const ParallelLoopBody &body,
                   double nstripes);

  void stop() { setWorkThreads(0); }

 private:
  bool setWorkThreads(int num);

 public:
  std::mutex mutex_notify_;
  std::condition_variable job_complete_;

 private:
  std::mutex pool_mutex_;
  int thread_num_;
  std::vector<std::shared_ptr<WorkerThread>> work_threads_;
  std::shared_ptr<ParallelJob> job_ = nullptr;
  const int active_wait_;
};

class ParallelForApiDefault : public ParallelForApi {
 public:
  ParallelForApiDefault() = default;

  virtual ~ParallelForApiDefault() = default;

  virtual void setThreadNum(int num);

  virtual int getThreadNum();

  virtual int parallelFor(const base::Range &range,
                          const ParallelLoopBody &body, double nstripes = -1.0);
};

}  // namespace thread_pool
}  // namespace nndeploy

#endif /* _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_H_ */
