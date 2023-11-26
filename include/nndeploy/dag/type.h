#ifndef _NNDEPLOY_DAG_TYPE_H_
#define _NNDEPLOY_DAG_TYPE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"

/**
 * @brief
 * node需要被用户继承，保持这块的用户代码逻辑尽量简单
 * 把复杂性放到edge和graph中去
 */

namespace nndeploy {
namespace dag {

enum ParallelType : int {
  kParallelTypeNone = 0x0001,
  kParallelTypeTask = 0x0001 << 1,
  kParallelTypePipeline = 0x0001 << 2,
  kParallelTypeAdapt = 0x0001 << 3,
};

enum EdgeType : int {
  kEdgeTypeFixed = 0x0001,
  kEdgeTypePipeline = 0x0001 << 1,
};

}  // namespace dag
}  // namespace nndeploy

#endif