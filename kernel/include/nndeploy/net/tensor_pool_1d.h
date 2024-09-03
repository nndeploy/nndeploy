

#ifndef _NNDEPLOY_NET_TENSOR_POOL_1D_H_
#define _NNDEPLOY_NET_TENSOR_POOL_1D_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

enum TensorPool1DSharedObjectType : int {
  kTensorPool1DSharedObjectTypeGreedyByBreadth = 0x0000,
  kTensorPool1DSharedObjectTypeGreedyBySize,
  kTensorPool1DSharedObjectTypeGreedyBySizeImprove,
  kTensorPool1DSharedObjectTypeNone,
};

class TensorPool1DSharedObject : public TensorPool {
 public:
  TensorPool1DSharedObject(device::Device *device,
                           std::vector<TensorWrapper *> &tensor_repository,
                           std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool1DSharedObject();

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

class TensorPool1DSharedObjectGreedyBySizeImprove
    : public TensorPool1DSharedObject {
 public:
  TensorPool1DSharedObjectGreedyBySizeImprove(
      device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool1DSharedObjectGreedyBySizeImprove();

  virtual base::Status allocate();
  virtual base::Status deallocate();

 private:
  std::vector<std::shared_ptr<Chunk>> chunks_;
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_1D_H_ */
