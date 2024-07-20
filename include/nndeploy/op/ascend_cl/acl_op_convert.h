
#ifndef _NNDEPLOY_OP_ASCEND_CL_ACL_OP_CONVERT_H_
#define _NNDEPLOY_OP_ASCEND_CL_ACL_OP_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"

namespace nndeploy {
namespace op {

class AclOpConvert {
 public:
  static base::DataType convertToDataType(const aclDataType &src);
  static aclDataType convertFromDataType(const base::DataType &src);

  static aclFormat convertFromDataType(const base::DataFormat &src);

  template <typename T>
  aclScalar *ConvertFromScalar(const base::Scalar<T> &src);

  aclTensor *ConvertType(const device::Tensor *src);

  aclTensorList *ConvertType(const std::vector<device::Tensor *> &src);
};

}  // namespace op
}  // namespace nndeploy

#endif
