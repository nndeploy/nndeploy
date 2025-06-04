#ifndef _NNDEPLOY_MATTING_MATTING_RESULT_H_
#define _NNDEPLOY_MATTING_MATTING_RESULT_H_

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

namespace nndeploy {
namespace matting {

class NNDEPLOY_CC_API MattingResult : public base::Param {
 public:
  MattingResult(){};
  ~MattingResult(){};

  std::vector<float> alpha;
  std::vector<float> foreground;
  std::vector<int64_t> shape;
  bool contain_foreground = false;
};

}  // namespace matting
}  // namespace nndeploy

#endif  // _NNDEPLOY_MATTING_MATTING_RESULT_H_