#ifndef _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_
#define _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/type.h"
#include "nndeploy/thread_pool/parallel.h"

namespace nndeploy {
namespace thread_pool {

class WorkerThread;
class ParallelJob;

class ThreadPool {
 public:
  static ThreadPool& getInstance() {
    static ThreadPool* instance = new ThreadPool;
    return *instance;
  }

  static void stop() { setWorkThreads(0); }

  void run(const base::Range& range, const ParallelLoopBody& body,
           double nstripes);

  size_t getNumOfThreads();

  void setNumOfThreads(unsigned n);

  ThreadPool();

  ~ThreadPool();

 private:
  bool setWorkThreads(unsigned int num);

 public:
  std::mutex mutex_notify_;
  std::condition_variable job_complete_;

 private:
  std::mutex pool_mutex_;
  unsigned int thread_num_;
  std::vector<std::shared_ptr<WorkerThread>> work_threads_;
  std::shared_ptr<ParallelJob> job_;
  const int active_wait_;
};

}  // namespace thread_pool
}  // namespace nndeploy

#endif /* _NNDEPLOY_THREAD_POOL_THREAD_POOL_H_ */
