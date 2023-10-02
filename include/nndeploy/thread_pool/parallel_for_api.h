#ifndef _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_H_
#define _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_H_

#include "nndeploy/thread_pool/parallel.h"

namespace nndeploy {
namespace thread_pool {

class ParallelForApi {
 public:
  ParallelForApi() = default;

  virtual ~ParallelForApi() = default;

  virtual void setThreadNum(int num) = 0;

  virtual int getThreadNum() = 0;

  virtual int parallelFor(const base::Range& range,
                          const ParallelLoopBody& body,
                          double nstripes = -1.0) = 0;
};

std::shared_ptr<ParallelForApi>& getParallelForApi(
    ParallelForApiType type = kParallelForApiTypeDefault);

}  // namespace thread_pool
}  // namespace nndeploy

#endif /* _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_H_ */
