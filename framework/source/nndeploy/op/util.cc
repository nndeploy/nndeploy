

#include "nndeploy/op/util.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"


namespace nndeploy {
namespace op {

// adjustNegativeAxes: Negative axes values are translated to the right axis in
// the positive range
void adjustNegativeAxes(int& axes, int rank) {
  axes = axes < 0 ? axes + rank : axes;
}

void adjustNegativeAxes(std::vector<int>& axes, int rank) {
  std::transform(axes.begin(), axes.end(), axes.begin(), [&](int axis) -> int {
    return axis < 0 ? axis + rank : axis;
  });
}

// checkAxesRange: Checks that values are within the range [-rank, rank)
bool checkAxesRange(int axes, int rank) {
  bool flag = true;
  if (axes < -rank || axes > (rank - 1)) {
    NNDEPLOY_LOGE("Unexpected axis value: %d.\n", axes);
    return false;
  }
  return flag;
}
bool checkAxesRange(std::vector<int>& axes, int rank) {
  bool flag = true;
  for (auto axis : axes) {
    if (axis < -rank || axis > (rank - 1)) {
      NNDEPLOY_LOGE("Unexpected axis value: %d.\n", axis);
      return false;
    }
  }
  return flag;
}

int32_t multiplyDims(const base::IntVector& shape, int from, int upto_exclusive) {
  int32_t dim = 1;
  for (int i = from; i < upto_exclusive; ++i) {
    dim = dim * shape[i];
  }
  return dim;
}

}  // namespace op
}  // namespace nndeploy