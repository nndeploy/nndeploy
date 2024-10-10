
#ifndef _NNDEPLOY_NET_TENSOR_POOL_H_
#define _NNDEPLOY_NET_TENSOR_POOL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/net/util.h"

namespace nndeploy {
namespace net {

// 只有激活值
struct TensorUsageRecord {
  TensorWrapper *tensor_wrapper_;
  size_t size_;
  std::array<int, 2> interval_;
  bool is_allocated_ = false;

  bool operator<(const TensorUsageRecord &other) const {
    return size_ < other.size_;
  }
};

struct OpBreadth {
  OpWrapper *op_wrapper_;
  std::vector<std::shared_ptr<TensorUsageRecord>> breadth_;
  size_t size_;

  bool operator<(const OpBreadth &other) const { return size_ < other.size_; }
};

struct Chunk {
  device::Buffer *buffer_;
  std::vector<std::array<int, 2>> intervals_;
};

class TensorPool {
 public:
  TensorPool(device::Device *device,
             std::vector<TensorWrapper *> &tensor_repository,
             std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool();

  virtual base::Status allocate() = 0;
  virtual base::Status deallocate() = 0;

 protected:
  device::Device *device_;
  base::IntVector config_ = base::IntVector();
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository;
};

std::vector<int> getOpOrderIndex(std::vector<OpWrapper *> &producers,
                                 std::vector<OpWrapper *> &consumers,
                                 std::vector<OpWrapper *> &op_repository);

bool isInterval(std::array<int, 2> &interval,
                std::vector<std::array<int, 2>> &intervals);

void tensorUsageRecordPrint(
    const std::vector<std::shared_ptr<TensorUsageRecord>>
        &tensor_usage_records);

void chunkPrint(const std::vector<std::shared_ptr<Chunk>> &chunks);

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_H_ */
