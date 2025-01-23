#ifndef _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_OFFSET_CALCULATE_BY_BREADTH_H_
#define _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_OFFSET_CALCULATE_BY_BREADTH_H_

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

class TensorPool1DOffsetCalculateGreedyByBreadth : public TensorPool1D {
 public:
  TensorPool1DOffsetCalculateGreedyByBreadth(
      device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool1DOffsetCalculateGreedyByBreadth();

  virtual base::Status allocate();
  virtual base::Status deallocate();

  /**
   * @brief 获取推理所需的内存大小
   *
   * @return int64_t
   */
  virtual int64_t getMemorySize();
  /**
   * @brief 设置推理所需的内存（推理内存由外部分配）
   *
   * @param buffer
   * @return base::Status
   */
  virtual base::Status setMemory(device::Buffer *buffer);

 private:
  std::vector<std::shared_ptr<TensorUsageRecord>>
      ordered_allocated_ids_;  // 已分配tensor
  int tensor_num_ = 0;
  int64_t total_consumption_ = -1;
  bool is_external_ = false;
  device::Buffer *mem_block_ = nullptr;
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_TENSOR_POOL_1D_OFFSET_CALCULATE_BY_BREADTH_H_ \
        */