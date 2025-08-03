
#ifndef _NNDEPLOY_NET_TENSOR_POOL_H_
#define _NNDEPLOY_NET_TENSOR_POOL_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/net/util.h"

/*
 * 内存复用
 * 1. 5种算法的实现
 * 2. 生命周期的优化
 * 3. 多模型共享内存的优化
 * 4. workspace的优化
 * 5. inplace算子的优化（TODO）
 * 6. 更精细的内存优化（基于多生命周期的优化）（TODO）
 */

namespace nndeploy {
namespace net {

enum TensorPoolType : int {
  kTensorPool1DSharedObjectTypeGreedyByBreadth,      // 正确
  kTensorPool1DSharedObjectTypeGreedyBySize,         // 正确
  kTensorPool1DSharedObjectTypeGreedyBySizeImprove,  // 正确
  kTensorPool1DOffsetCalculateTypeGreedyBySize,      // 正确
  kTensorPool1DOffsetCalculateTypeGreedyByBreadth,   // 正确
  kTensorPool1DNone,
};

// 只有激活值
struct TensorUsageRecord {
  TensorWrapper *tensor_wrapper_;
  size_t size_;
  std::array<int, 2> interval_;
  int offset_ = -1;  // 初始化offset为-1
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
  // 共享指针 buffer->getData()
  device::Buffer *buffer_;
  std::vector<std::array<int, 2>> intervals_;
};

struct Offset {
  int offset_;
  size_t size_;
  std::vector<std::shared_ptr<TensorUsageRecord>> tensor_usage_records_;
};

class TensorPool {
 public:
  TensorPool(device::Device *device,
             std::vector<TensorWrapper *> &tensor_repository,
             std::vector<OpWrapper *> &op_repository);
  virtual ~TensorPool();

  virtual base::Status setIsExternal(bool is_external);

  virtual base::Status allocate() = 0;
  virtual base::Status deallocate() = 0;

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

 protected:
  device::Device *device_;
  base::IntVector config_ = base::IntVector();
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;
  bool is_external_ = false;
};

/**
 * @brief TensorPool的创建类
 *
 */
class TensorPoolCreator {
 public:
  virtual ~TensorPoolCreator() {};
  virtual TensorPool *createTensorPool(
      device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository) = 0;
};

/**
 * @brief TensorPool的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeTensorPoolCreator : public TensorPoolCreator {
  virtual TensorPool *createTensorPool(
      device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository) {
    return new T(device, tensor_repository, op_repository);
  }
};

/**
 * @brief Get the Global TensorPool Creator Map object
 *
 * @return std::map<TensorPoolType, std::shared_ptr<TensorPoolCreator>>&
 */
std::map<TensorPoolType, std::shared_ptr<TensorPoolCreator>> &
getGlobalTensorPoolCreatorMap();

/**
 * @brief TensorPool的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeTensorPoolRegister {
 public:
  explicit TypeTensorPoolRegister(TensorPoolType type) {
    getGlobalTensorPoolCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a TensorPool object
 *
 * @param type
 * @param device
 * @param tensor_repository
 * @param op_repository
 * @return TensorPool*
 */
extern NNDEPLOY_CC_API TensorPool *createTensorPool(
    TensorPoolType type, device::Device *device,
    std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository);

std::vector<int> getOpOrderIndex(std::vector<OpWrapper *> &producers,
                                 std::vector<OpWrapper *> &consumers,
                                 std::vector<OpWrapper *> &op_repository);

bool isInterval(std::array<int, 2> &interval,
                std::vector<std::array<int, 2>> &intervals);

void tensorUsageRecordPrint(
    const std::vector<std::shared_ptr<TensorUsageRecord>>
        &tensor_usage_records);

void chunkPrint(const std::vector<std::shared_ptr<Chunk>> &chunks);

extern NNDEPLOY_CC_API std::string tensorPoolTypeToString(TensorPoolType type);
extern NNDEPLOY_CC_API TensorPoolType stringToTensorPoolType(
    const std::string &src);

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_TENSOR_POOL_H_ */
