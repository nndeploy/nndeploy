
#ifndef _NNDEPLOY_OP_ASCEND_CL_ACL_OP_CONVERT_H_
#define _NNDEPLOY_OP_ASCEND_CL_ACL_OP_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/half.hpp"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/acl_op_include.h"
#include "nndeploy/op/ascend_cl/acl_op_util.h"

namespace nndeploy {
namespace op {

class AclOpConvert {
 public:
  static base::DataType convertToDataType(const aclDataType &src);
  static aclDataType convertFromDataType(const base::DataType &src);

  static aclFormat convertFromDataFormat(const base::DataFormat &src);
  static base::IntVector inferShape(const aclFormat &format,
                                    base::IntVector shape);

  static std::vector<int64_t> convertFromShape(base::IntVector &src);

  template <typename T>
  static aclScalar *convertFromScalar(const base::Scalar<T> &src);

  template <typename T>
  static aclScalar *convertFromScalar(float src);

  static aclScalar *convertFromScalar(float src,
                                      const base::DataType &data_type);
  // template <typename T>
  // aclScalarList *convertFromScalar(const std::vector<base::Scalar<T>> &src);

  static aclIntArray *convertFromIntVector(const std::vector<int> &src);
  static aclFloatArray *convertFromFloatVector(const std::vector<float> &src);
  // aclBoolArray *convertFromBoolVector(const std::vector<bool> &src);
  // aclFp16Array *convertFromFp16Vector(const std::vector<half_float::half>
  // &src); aclBf16Array *convertFromBfp16Vector(const
  // std::vector<base::bfp16_t> &src);

  static aclTensor *convertFromTensor(const device::Tensor *src);
  static aclTensorList *convertFromTensor(
      const std::vector<device::Tensor *> &src);
};

}  // namespace op
}  // namespace nndeploy

#endif
