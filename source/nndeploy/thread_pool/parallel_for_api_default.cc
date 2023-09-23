
#include "nndeploy/thread_pool/parallel_for_api.h"

namespace nndeploy {
namespace thread_pool {

void ParallelForApiDefault::setThreadNum(int num) {
  ThreadPool::getinstance().setThreadNum(num);
}

int ParallelForApiDefault::getThreadNum() const {
  return ThreadPool::getinstance().getThreadNum();
}

int ParallelForApiDefault::parallelFor(const Range& range,
                                       const ParallelTask& task, int nstripes) {
  ThreadPool::getinstance().run(range, task, nstripes);
  return 0;
}

}  // namespace thread_pool
}  // namespace nndeploy