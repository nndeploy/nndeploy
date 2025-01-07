#ifndef _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_BASE_H_
#define _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_BASE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class TensorPool1D : public TensorPool {
 public:
  TensorPool1D(device::Device *device,
               std::vector<TensorWrapper *> &tensor_repository,
               std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool1D();

  base::Status initTensorUsageRecord();
  base::Status deinitTensorUsageRecord();

  base::Status initOpBreadth();
  base::Status deinitOpBreadth();

  base::Status initPositionalMaximum();
  base::Status deinitPositionalMaximum();

 protected:
  std::vector<std::shared_ptr<TensorUsageRecord>> tensor_usage_records_;
  std::vector<std::shared_ptr<OpBreadth>> op_breadths_;
  std::vector<size_t> positional_maximum_;
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_1D_BASE_H_ */