
#include "nndeploy/thread_pool/parallel.h"

namespace nndeploy {
namespace thread_pool {

void setThreadNum(int num) { return; }

int getThreadNum() { return 1; }

void parallelFor(const base::Range& range, const ParallelLoopBody& body,
                 double nstripes) {
  return;
}

}  // namespace thread_pool
}  // namespace nndeploy
