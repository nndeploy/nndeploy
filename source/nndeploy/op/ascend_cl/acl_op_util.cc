
#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"

namespace nndeploy {
namespace op {

std::string getTensorDescString(const aclTensorDesc* desc) {
  auto data_type = aclGetTensorDescType(desc);
  auto origin_format = aclGetTensorDescFormat(desc);  // origin format

  std::stringstream ss;
  ss << "TensorDesc: data_type = " << data_type
     << ", origin_format = " << origin_format << ", origin_dims = [";

  size_t rank = aclGetTensorDescNumDims(desc);
  for (auto i = 0; i < rank; ++i) {
    int64_t dim_size = -1;
    PADDLE_ENFORCE_NPU_SUCCESS(aclGetTensorDescDimV2(desc, i, &dim_size));
    ss << dim_size;
    if (i < rank - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::string getOpDescString(std::vector<aclTensorDesc*> descs,
                            const std::string msg);

std::string getOpInfoString(std::vector<aclTensorDesc*> descs,
                            std::vector<aclDataBuffer*> buffs,
                            const std::string msg);

template <typename T>
aclDataType aclDataTypeOf() {
  return ACL_FLOAT;
}
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<float>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<double>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<uint8_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<uint16_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<uint32_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<uint64_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<int8_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<int16_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<int32_t>();
template <>
NNDEPLOY_CC_API aclDataType aclDataTypeOf<int64_t>();

}  // namespace op
}  // namespace nndeploy
