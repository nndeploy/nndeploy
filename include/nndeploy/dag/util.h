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

bool checkEdge(const std::vector<Edge *> &src_edges,
               const std::vector<Edge *> &dst_edges);

/**
 * @brief 对vector插入不在vector中的元素，即类似集合的作用
 * @tparam T
 * @param  vec              My Param doc
 * @param  val              My Param doc
 */
template <typename T>
void insertUnique(std::vector<T> &vec, const T &val) {
  if (std::find(vec.begin(), vec.end(), val) == vec.end()) {
    vec.emplace_back(val);
  }
}
}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_UTIL_H_ */
