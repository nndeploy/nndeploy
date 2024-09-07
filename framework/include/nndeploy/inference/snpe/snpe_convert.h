#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_CONVERT_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/snpe/snpe_include.h"
#include "nndeploy/inference/snpe/snpe_inference_param.h"

namespace nndeploy {
namespace inference {

class SnpeConvert {
 public:
  SnpeConvert();
  ~SnpeConvert();

  static base::DataType convertToDataType(SnpeBuffer_Type_t &src);

  static base::DataFormat convertToDataFormat();

  static base::IntVector convertToShape(const zdl::DlSystem::Dimension *dims,
                                        size_t rank);
};

}  // namespace inference
}  // namespace nndeploy

#endif