
#ifndef _NNDEPLOY_FORWARD_UTIL_H_
#define _NNDEPLOY_FORWARD_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API OpWrapper {
 public:
  bool is_external_;
  op::Op *op_;
  std::string name_;
  std::vector<OpWrapper *> predecessors_;
  std::vector<OpWrapper *> successors_;
  base::NodeColorType color_ = base::kNodeColorWhite;
};

class NNDEPLOY_CC_API TensorWrapper {
 public:
  bool is_external_;
  device::Tensor *edge_;
  std::string name_;
  std::vector<OpWrapper *> producers_;
  std::vector<OpWrapper *> consumers_;
};

NNDEPLOY_CC_API Tensor *getTensor(std::vector<TensorWrapper *> &edge_repository,
                                  const std::string &edge_name);
NNDEPLOY_CC_API TensorWrapper *findTensorWrapper(
    std::vector<TensorWrapper *> &edge_repository,
    const std::string &edge_name);
NNDEPLOY_CC_API TensorWrapper *findTensorWrapper(
    std::vector<TensorWrapper *> &edge_repository, device::Tensor *edge);
NNDEPLOY_CC_API std::vector<TensorWrapper *> findStartTensors(
    std::vector<TensorWrapper *> &edge_repository);
NNDEPLOY_CC_API std::vector<TensorWrapper *> findEndTensors(
    std::vector<TensorWrapper *> &edge_repository);

NNDEPLOY_CC_API op::Op *getNode(std::vector<OpWrapper *> &op_repository,
                                const std::string &name);
NNDEPLOY_CC_API OpWrapper *findOpWrapper(
    std::vector<OpWrapper *> &op_repository, const std::string &name);
NNDEPLOY_CC_API OpWrapper *findOpWrapper(
    std::vector<OpWrapper *> &op_repository, op::Op *node);
NNDEPLOY_CC_API std::vector<OpWrapper *> findStartNodes(
    std::vector<OpWrapper *> &op_repository);
NNDEPLOY_CC_API std::vector<OpWrapper *> findEndNodes(
    std::vector<OpWrapper *> &op_repository);

NNDEPLOY_CC_API base::Status setColor(std::vector<OpWrapper *> &op_repository,
                                      base::NodeColorType color);

base::Status dumpDag(std::vector<TensorWrapper *> &edge_repository,
                     std::vector<OpWrapper *> &op_repository,
                     std::vector<device::Tensor *> &graph_inputs,
                     std::vector<device::Tensor *> &graph_outputs,
                     const std::string &name, std::ostream &oss);

std::vector<OpWrapper *> checkUnuseNode(
    std::vector<OpWrapper *> &op_repository);
std::vector<TensorWrapper *> checkUnusedevice::Tensor(
    std::vector<OpWrapper *> &op_repository,
    std::vector<TensorWrapper *> &edge_repository);

base::Status topoSortBFS(std::vector<OpWrapper *> &op_repository,
                         std::vector<OpWrapper *> &topo_sort_node);

base::Status topoSortDFS(std::vector<OpWrapper *> &op_repository,
                         std::vector<OpWrapper *> &topo_sort_node);

base::Status topoSort(std::vector<OpWrapper *> &op_repository,
                      base::TopoSortType topo_sort_type,
                      std::vector<OpWrapper *> &topo_sort_node);

bool checkdevice::Tensor(const std::vector<device::Tensor *> &src_edges,
                         const std::vector<device::Tensor *> &dst_edges);

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

}  // namespace forward
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_UTIL_H_ */
