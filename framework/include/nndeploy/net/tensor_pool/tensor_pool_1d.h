

#ifndef _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_H_
#define _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_H_

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

class TensorPool1DSharedObjectGreedyByBreadth
    : public TensorPool1DSharedObject {
 public:
  TensorPool1DSharedObjectGreedyByBreadth(
      device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool1DSharedObjectGreedyByBreadth();

  virtual base::Status allocate();
  virtual base::Status deallocate();

 private:
  std::vector<std::shared_ptr<Chunk>> chunks_;
  std::unordered_map<std::shared_ptr<Chunk>, size_t>
      chunk_sizes_;  // 由于Chunk可能扩容，先预存size，在分配算法结束后统一开辟

  std::unordered_map<std::shared_ptr<Chunk>,
                     std::set<std::shared_ptr<TensorUsageRecord>>>
      chunk_schedules_;  // 记录Chunk由哪些tensor共享
  std::set<std::shared_ptr<TensorUsageRecord>> assigned_tensors_;
  // 记录已经处理过的tensor：由于延迟开辟内存，无法根据tensor的allocated属性判断
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_1D_H_ */