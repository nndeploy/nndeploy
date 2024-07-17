
#ifndef _NNDEPLOY_NET_ASCEND_CL_ASCEND_CL_SESSION_H_
#define _NNDEPLOY_NET_ASCEND_CL_ASCEND_CL_SESSION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/net/session.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API AscendCLSession : public Session {
 public:
  AscendCLSession(){};
  virtual ~AscendCLSession(){};

  virtual base::Status init(std::vector<TensorWrapper *> &tensor_repository,
                            std::vector<OpWrapper *> &op_repository,
                            std::map<std::string, device::Tensor *> weights);
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();
};

}  // namespace net
}  // namespace nndeploy

#endif
