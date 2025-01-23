#ifndef _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_SHARED_OBJECT_BY_BREADTH_H_
#define _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_SHARED_OBJECT_BY_BREADTH_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/tensor_pool/tensor_pool_1d_base.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

class TensorPool1DSharedObjectGreedyByBreadth : public TensorPool1D {
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

#endif /* _NNDEPLOY_NET_TENSOR_POOL_1D_SHARED_OBJECT_BY_BREADTH_H_ */