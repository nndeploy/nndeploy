#ifndef _NNDEPLOY_OP_X86_OP_CONVERT_H_
#define _NNDEPLOY_OP_X86_OP_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/half.hpp"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/x86/op_include.h"

namespace nndeploy {
namespace op {
class X86OpConvert {
 public:
  static base::DataType convertToDataType(const dnnl::memory::data_type &src);
  static dnnl::memory::data_type convertFromDataType(const base::DataType &src);

  static dnnl::memory::dims convertFromShape(const base::IntVector &src);

  static base::DataFormat convertToDataFormat(
      const dnnl::memory::format_tag &src);
  static dnnl::memory::format_tag convertFromDataFormat(
      const base::DataFormat &src);
};
}  // namespace op
}  // namespace nndeploy

#endif