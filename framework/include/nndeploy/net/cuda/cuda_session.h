
#ifndef _NNDEPLOY_NET_CUDA_CUDA_SESSION_H_
#define _NNDEPLOY_NET_CUDA_CUDA_SESSION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/cuda/cuda_device.h"
#include "nndeploy/net/session.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API CudaSession : public base::NonCopyable {
 public:
  CudaSession() {};
  virtual ~CudaSession() {};

  virtual base::Status init(std::vector<TensorWrapper *> &tensor_repository,
                            std::vector<OpWrapper *> &op_repository) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

  virtual base::Status preRun() = 0;
  virtual base::Status run() = 0;
  virtual base::Status postRun() = 0;
};

}  // namespace net
}  // namespace nndeploy

#endif
