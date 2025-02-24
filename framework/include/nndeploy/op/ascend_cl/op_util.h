#ifndef _NNDEPLOY_OP_ASCEND_CL_OP_UTIL_H_
#define _NNDEPLOY_OP_ASCEND_CL_OP_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/op_include.h"

namespace nndeploy {
namespace op {

std::string getDataBufferString(const aclDataBuffer* buf);

std::string getTensorDescString(const aclTensorDesc* desc);

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

// 检查aclnnStatus
#define CHECK_ACLNN_STATUS(func_call)                                 \
  do {                                                                \
    aclnnStatus status = (func_call);                                 \
    if (status != ACL_SUCCESS) {                                      \
      NNDEPLOY_LOGE(#func_call " failed, error code: %d.\n", status); \
      return base::kStatusCodeErrorOpAscendCL;                        \
    }                                                                 \
  } while (0)

int64_t getAclOpShapeSize(const std::vector<int64_t>& shape);

std::vector<int64_t> getAclOpStrides(const std::vector<int64_t>& shape);

}  // namespace op
}  // namespace nndeploy

#endif
