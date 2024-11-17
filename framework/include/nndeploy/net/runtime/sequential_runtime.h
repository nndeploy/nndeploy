
#ifndef _NNDEPLOY_NET_RUNTIME_SEQUENTIAL_RUNTIME_H_
#define _NNDEPLOY_NET_RUNTIME_SEQUENTIAL_RUNTIME_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/device/device.h"
#include "nndeploy/net/runtime.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API SequentialRuntime : public Runtime {
 public:
  SequentialRuntime(const base::DeviceType &device_type);
  virtual ~SequentialRuntime();

  virtual base::Status init(
      std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
      base::ShapeMap max_shape,
      TensorPoolType tensor_pool_type =
          kTensorPool1DSharedObjectTypeGreedyBySizeImprove);
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();


 private:
  bool workspace_is_external_ = false;  // workspace是否是外部传入
  uint64_t workspace_size_ = 0;         // workspace大小
  void *workspace_ = nullptr;           // op的workspace
};

}  // namespace net
}  // namespace nndeploy

#endif
