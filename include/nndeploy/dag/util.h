#ifndef _NNDEPLOY_DAG_UTIL_H_
#define _NNDEPLOY_DAG_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/type.h"

namespace nndeploy {
namespace dag {

bool checkEdge(const std::vector<Edge*>& edges,
               const std::vector<Edge*>& check_edges);

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_UTIL_H_ */
