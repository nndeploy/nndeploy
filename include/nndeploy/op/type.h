
#ifndef _NNDEPLOY_OP_TYPE_H_
#define _NNDEPLOY_OP_TYPE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"

namespace nndeploy {
namespace op {

// GEMMA的算子集合
enum NNOpType : int {
  kNNOpTypeForward = 0x0000,

  kNNOpTypeGemv,
  kNNOpTypeGelu,
  kNNOpTypeRMSNorm,
  kNNOpTypeNone,
}

}  // namespace op
}  // namespace nndeploy

#endif /* _NNDEPLOY_OP_TYPE_H_ */
