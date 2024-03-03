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

namespace nndeploy {
namespace dag {

enum ParallelType : int {
  kParallelTypeNone = 0x0001,
  kParallelTypeSequential = 0x0001 << 1,
  kParallelTypeTask = 0x0001 << 2,
  kParallelTypePipeline = 0x0001 << 3,
};

enum EdgeType : int {
  kEdgeTypeFixed = 0x0001,
  kEdgeTypePipeline = 0x0001 << 1,
};

enum NodeColorType : int {
  kNodeColorWhite = 0x0000,
  kNodeColorGray,
  kNodeColorBlack
};

enum TopoSortType : int { kTopoSortTypeBFS = 0x0000, kTopoSortTypeDFS };

}  // namespace dag
}  // namespace nndeploy

#endif