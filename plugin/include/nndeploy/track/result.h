#ifndef _NNDEPLOY_TRACK_RESULT_H_
#define _NNDEPLOY_TRACK_RESULT_H_

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
namespace track {

class NNDEPLOY_CC_API MOTResult : public base::Param {
 public:
  MOTResult() {}
  virtual ~MOTResult() {}

  /**
   * @brief The tracking object for an input images
   */
  std::vector<std::array<int, 4>> boxes;

  /**
   * @brief The tracking object ids
   */
  std::vector<int> ids;

  /**
   * @brief The confidence for all the tracking objects
   */
  std::vector<float> scores;

  /**
   * @brief The classify label for all the tracking objects
   */
  std::vector<int> class_ids;
};

}  // namespace track
}  // namespace nndeploy

#endif