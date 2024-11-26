
#ifndef _NNDEPLOY_SEGMENT_UTIL_H_
#define _NNDEPLOY_SEGMENT_UTIL_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/segment/result.h"

namespace nndeploy {
namespace segment {

NNDEPLOY_CC_API device::Tensor *convertVectorToTensor(
    std::vector<float> &data, std::vector<int> dst_shape,
    device::Device *device, base::DataFormat data_format, std::string name);

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_COMMON_H_ */
