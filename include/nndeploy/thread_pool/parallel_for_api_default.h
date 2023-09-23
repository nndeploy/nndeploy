
#ifndef _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_DEFAULT_H_
#define _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_DEFAULT_H_

#include "nndeploy/thread_pool/parallel.h"

namespace nndeploy {
namespace thread_pool {

class ParallelForApiDefault : public ParallelForApi {
 public:
  ParallelForApiDefault() = default;

  virtual ~ParallelForApiDefault() = default;

  virtual void setThreadNum(int num);

  virtual int getThreadNum();

  virtual int parallelFor(const base::Range& range,
                          const ParallelLoopBody& body, double nstripes = -1.0);
};

}  // namespace thread_pool
}  // namespace nndeploy

#endif /* _NNDEPLOY_THREAD_POOL_PARALLEL_FOR_API_H_ */
