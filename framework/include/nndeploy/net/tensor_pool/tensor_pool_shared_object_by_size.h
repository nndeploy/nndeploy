#ifndef _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_SHARED_OBJECT_BY_SIZE_H_
#define _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_SHARED_OBJECT_BY_SIZE_H_

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
#include "nndeploy/net/tensor_pool/tensor_pool_1d_base.h"

namespace nndeploy {
namespace net {

class TensorPool1DSharedObjectGreedyBySize
    : public TensorPool1D {
 public:
  TensorPool1DSharedObjectGreedyBySize(
      device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool1DSharedObjectGreedyBySize();



  virtual base::Status allocate();
  virtual base::Status deallocate();

 private:
  std::vector<std::shared_ptr<Chunk>> chunks_;
};


}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_SHARED_OBJECT_BY_SIZE_H_ */