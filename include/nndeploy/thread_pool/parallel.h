#ifndef _NNDEPLOY_THREAD_POOL_PARALLEL_H_
#define _NNDEPLOY_THREAD_POOL_PARALLEL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/type.h"

namespace nndeploy {
namespace thread_pool {

enum ParallelForApiType : int {
  kParallelForApiTypeDefault = 0x0000,
};

/**
 * @brief Base class for parallel data processors
 */
class NNDEPLOY_CC_API ParallelLoopBody {
 public:
  virtual ~ParallelLoopBody() {}
  virtual void operator()(const base::Range& range) const = 0;
};

extern NNDEPLOY_CC_API int defaultNumberOfThreads();

extern NNDEPLOY_CC_API void setThreadNum(int num);

extern NNDEPLOY_CC_API int getThreadNum();

/**
 * @brief Parallel data processor
 */
extern NNDEPLOY_CC_API void parallelFor(const base::Range& range,
                                        const ParallelLoopBody& body,
                                        double nstripes = -1.0);

}  // namespace thread_pool
}  // namespace nndeploy

#endif /* _NNDEPLOY_THREAD_POOL_PARALLEL_H_ */
