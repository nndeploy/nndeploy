

#ifndef _NNDEPLOY_OP_UTIL_H_
#define _NNDEPLOY_OP_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace op {

// adjustNegativeAxes: Negative axes values are translated to the right axis in
// the positive range
void adjustNegativeAxes(int& axes, int rank);

void adjustNegativeAxes(std::vector<int>& axes, int rank);

// checkAxesRange: Checks that values are within the range [-rank, rank)
bool checkAxesRange(int axes, int rank);
bool checkAxesRange(std::vector<int>& axes, int rank);

}  // namespace op
}  // namespace nndeploy

#endif /* _NNDEPLOY_OP_UTIL_H_ */