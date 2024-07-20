#ifndef _NNDEPLOY_OP_ASCEND_CL_ACL_OP_CONVERT_H_
#define _NNDEPLOY_OP_ASCEND_CL_ACL_OP_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"

namespace nndeploy {
namespace op {

std::string GetDataBufferString(const aclDataBuffer* buf);

std::string GetTensorDescString(const aclTensorDesc* desc);

std::string GetOpDescString(std::vector<aclTensorDesc*> descs,
                            const std::string msg);

std::string GetOpInfoString(std::vector<aclTensorDesc*> descs,
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

#endif
