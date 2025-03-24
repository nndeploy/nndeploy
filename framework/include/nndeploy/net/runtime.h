
#ifndef _NNDEPLOY_NET_RUNTIME_H_
#define _NNDEPLOY_NET_RUNTIME_H_

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

class Runtime;

class PipelineTensor {
 public:
  PipelineTensor() {};
  virtual ~PipelineTensor() {};
  std::vector<device::Tensor *> tensors_;
  std::vector<Runtime *> producers_;
  std::vector<Runtime *> consumers_;

  // 添加互斥锁和条件变量，用于同步不同阶段之间的数据传递
  std::mutex mutex_;
  std::condition_variable cv_;
  std::map<Runtime *, int> current_index_;

  void push(device::Tensor *tensor) {
    NNDEPLOY_LOGI("tensor name %s\n", tensor->getName().c_str());
    std::lock_guard<std::mutex> lock(mutex_);
    tensors_.push_back(tensor);
    cv_.notify_one();
  }
  device::Tensor *pop(Runtime *runtime) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, runtime]() {
      return current_index_[runtime] < tensors_.size();
    });
    device::Tensor *tensor = tensors_[current_index_[runtime]];
    current_index_[runtime]++;
    return tensor;
  }
};

class NNDEPLOY_CC_API Runtime : public base::NonCopyable {
 public:
  Runtime(const base::DeviceType &device_type) : device_type_(device_type) {};
  virtual ~Runtime() {
    if (!is_external_stream_ && stream_ != nullptr) {
      device::destroyStream(stream_);
      stream_ = nullptr;
    }
  };

  void setStream(device::Stream *stream);
  device::Stream *getStream();

  virtual base::Status init(
      std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
      base::ShapeMap max_shape,
      TensorPoolType tensor_pool_type =
          kTensorPool1DSharedObjectTypeGreedyBySizeImprove) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

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

  virtual base::Status preRun() = 0;
  virtual base::Status run() = 0;
  virtual base::Status postRun() = 0;

 protected:
  base::DeviceType device_type_;
  /**
   * @brief op的stream
   * note: 当stream为外部传入时，is_external_stream_为true
   */
  bool is_external_stream_ = false;
  device::Stream *stream_ = nullptr;
  TensorPoolType tensor_pool_type_ =
      kTensorPool1DOffsetCalculateTypeGreedyByBreadth;
  TensorPool *tensor_pool_;
  bool is_dynamic_shape_ = false;                // 是否是动态shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;
};

/**
 * @brief Runtime的创建类
 *
 */
class RuntimeCreator {
 public:
  virtual ~RuntimeCreator() {};

  virtual Runtime *createRuntime(const base::DeviceType &device_type,
                                 base::ParallelType parallel_type) = 0;
};

/**
 * @brief Runtime的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeRuntimeCreator : public RuntimeCreator {
  virtual Runtime *createRuntime(const base::DeviceType &device_type,
                                 base::ParallelType parallel_type) {
    auto Runtime = new T(device_type);
    return Runtime;
  }
};

/**
 * @brief Get the Global Runtime Creator Map object
 *
 * @return std::map<ExecutorType, std::map<const std::string &,
 * std::shared_ptr<RuntimeCreator>>>&
 */
std::map<base::ParallelType, std::shared_ptr<RuntimeCreator>> &
getGlobalRuntimeCreatorMap();

/**
 * @brief Runtime的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeRuntimeRegister {
 public:
  explicit TypeRuntimeRegister(base::ParallelType parallel_type) {
    getGlobalRuntimeCreatorMap()[parallel_type] = std::shared_ptr<T>(new T());
  }
};

Runtime *createRuntime(const base::DeviceType &device_type,
                       base::ParallelType parallel_type);

}  // namespace net
}  // namespace nndeploy

#endif
