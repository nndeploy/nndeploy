
#ifndef _NNDEPLOY_FORWARD_RUNTIME_H_
#define _NNDEPLOY_FORWARD_RUNTIME_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/forward/util.h"

namespace nndeploy {
namespace forward {

class NNDEPLOY_CC_API Runtime : public base::NonCopyable {
 public:
  Runtime(){};
  virtual ~Runtime(){};

  virtual base::Status init(std::vector<TensorWrapper *> &tensor_repository,
                            std::vector<OpWrapper *> &op_repository) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status run() = 0;
};

}  // namespace forward
}  // namespace nndeploy

#endif
