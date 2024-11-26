#include "nndeploy/net/optimizer.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace net {

NNDEPLOY_API_PYBIND11_MODULE("net", m) {
  py::enum_<OptPassType>(m, "OptPassType")
      .value("kOptPassTypeFuseConvBias", OptPassType::kOptPassTypeFuseConvBias)
      .value("kOptPassTypeFuseConvBatchNorm",
             OptPassType::kOptPassTypeFuseConvBatchNorm)
      .value("kOptPassTypeFuseConvRelu", OptPassType::kOptPassTypeFuseConvRelu)
      .value("kOptPassTypeEliminateCommonSubexpression",
             OptPassType::kOptPassTypeEliminateCommonSubexpression)
      .value("kOptPassTypeEliminateDeadOp",
             OptPassType::kOptPassTypeEliminateDeadOp)
      .export_values();  // 这一步是可选的，它会导出枚举值到Python的命名空间中
}

}  // namespace net
}  // namespace nndeploy