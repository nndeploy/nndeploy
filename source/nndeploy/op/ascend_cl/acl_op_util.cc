
#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"

namespace nndeploy {
namespace op {

std::string getDataBufferString(const aclDataBuffer* buf) {
  auto size = aclGetDataBufferSizeV2(buf);
  auto addr = aclGetDataBufferAddr(buf);
  auto numel = size / sizeof(float);
  std::vector<float> cpu_data(numel, 0);
  aclrtMemcpy(cpu_data.data(), size, addr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  std::stringstream ss;
  for (auto value : cpu_data) {
    ss << value << ",";
  }
  ss << "]";
  return ss.str();
}

std::string getTensorDescString(const aclTensorDesc* desc) {
  auto data_type = aclGetTensorDescType(desc);
  auto origin_format = aclGetTensorDescFormat(desc);  // origin format

  std::stringstream ss;
  ss << "TensorDesc: data_type = " << data_type
     << ", origin_format = " << origin_format << ", origin_dims = [";

  size_t rank = aclGetTensorDescNumDims(desc);
  for (auto i = 0; i < rank; ++i) {
    int64_t dim_size = -1;
    aclGetTensorDescDimV2(desc, i, &dim_size);
    ss << dim_size;
    if (i < rank - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::string getOpDescString(std::vector<aclTensorDesc*> descs,
                            const std::string msg) {
  std::stringstream ss;
  for (auto i = 0; i < descs.size(); ++i) {
    ss << " - " << msg << "[" << std::to_string(i)
       << "]: ";  // Input[i] or Output[i]
    ss << getTensorDescString(descs[i]) << "\n";
  }
  return ss.str();
}

std::string getOpInfoString(std::vector<aclTensorDesc*> descs,
                            std::vector<aclDataBuffer*> buffs,
                            const std::string msg) {
  if (descs.size() != buffs.size()) {
    NNDEPLOY_LOGE("descs.size()[%ld] != buffs.size()[%ld].\n", descs.size(),
                 buffs.size());
    return "";
  }
  std::stringstream ss;
  for (auto i = 0; i < descs.size(); ++i) {
    ss << msg << "[" << std::to_string(i) << "]: ";  // Input[i] or Output[i]
    ss << getTensorDescString(descs[i]) << "\n";
    ss << getDataBufferString(buffs[i]) << "\n";
  }
  return ss.str();
}

template <typename T>
aclDataType aclDataTypeOf() {
  return ACL_FLOAT;
}
template <>
aclDataType aclDataTypeOf<float>() {
  return ACL_FLOAT;
}
template <>
aclDataType aclDataTypeOf<double>() {
  return ACL_DOUBLE;
}
template <>
aclDataType aclDataTypeOf<uint8_t>() {
  return ACL_UINT8;
}
template <>
aclDataType aclDataTypeOf<uint16_t>() {
  return ACL_UINT16;
}
template <>
aclDataType aclDataTypeOf<uint32_t>() {
  return ACL_UINT32;
}
template <>
aclDataType aclDataTypeOf<uint64_t>() {
  return ACL_UINT64;
}
template <>
aclDataType aclDataTypeOf<int8_t>() {
  return ACL_INT8;
}
template <>
aclDataType aclDataTypeOf<int16_t>() {
  return ACL_INT16;
}
template <>
aclDataType aclDataTypeOf<int32_t>() {
  return ACL_INT32;
}
template <>
aclDataType aclDataTypeOf<int64_t>() {
  return ACL_INT64;
}

}  // namespace op
}  // namespace nndeploy
