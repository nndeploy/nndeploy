
#ifndef _NNDEPLOY_INFERENCE_MDC_MDC_CONVERT_H_
#define _NNDEPLOY_INFERENCE_MDC_MDC_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/mdc/mdc_include.h"
#include "nndeploy/inference/mdc/mdc_inference_param.h"

namespace nndeploy {
namespace inference {

class MdcConvert {
 public:
  static base::DataType convertToDataType(const aclDataType &src);
  static aclDataType convertFromDataType(const base::DataType &src);

  static base::DataFormat getDataFormatByShape(const base::IntVector &src);

  static base::IntVector convertToShape(std::vector<int64_t> &src, base::IntVector max_shape = base::IntVector());
  static std::vector<int64_t> convertFromShape(const base::IntVector &src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
