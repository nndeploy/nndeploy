#ifndef _NNDEPLOY_SOURCE_THREAD_PARALLEL_H_
#define _NNDEPLOY_SOURCE_THREAD_PARALLEL_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/type.h"

namespace nndeploy {
namespace thread {

/**
 * @brief Base class for parallel data processors
 */
class NNDEPLOY_CC_API ParallelLoopBody {
 public:
  virtual ~ParallelLoopBody() {}
  virtual void operator()(const base::Range& range) const = 0;
};

/**
 * @brief Parallel data processor
 */
extern NNDEPLOY_CC_API void parallelFor(const base::Range& range,
                                        const ParallelLoopBody& body,
                                        double nstripes = -1.0);

}  // namespace thread
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_THREAD_PARALLEL_H_ */
