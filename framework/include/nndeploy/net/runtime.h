
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

// Each queue entry owns one immutable frame. A consumer keeps its lease while
// copying/reading, independently of queue reclamation by other consumers.
class PipelineTensor {
 public:
  using Packet = std::shared_ptr<device::Tensor>;
  std::deque<Packet> tensors_;
  std::vector<Runtime *> producers_, consumers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::map<Runtime *, size_t> current_index_;
  bool is_finish_ = false;

  void push(Packet tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_finish_) tensors_.push_back(std::move(tensor));
    cv_.notify_all();
  }
  // Compatibility for borrowed tensors; the pipeline itself publishes owners.
  void push(device::Tensor *tensor) { push(Packet(tensor, [](device::Tensor *) {})); }
  void setFinish() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_finish_ = true;
    cv_.notify_all();
  }
  Packet popShared(Runtime *runtime) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!current_index_.count(runtime)) return nullptr;
    cv_.wait(lock, [&] { return is_finish_ || current_index_[runtime] < tensors_.size(); });
    if (is_finish_) return nullptr;
    auto result = tensors_[current_index_[runtime]++];
    size_t consumed = tensors_.size();
    for (const auto &cursor : current_index_) consumed = std::min(consumed, cursor.second);
    for (size_t i = 0; i < consumed; ++i) tensors_.pop_front();
    for (auto &cursor : current_index_) cursor.second -= consumed;
    return result;
  }
  device::Tensor *pop(Runtime *runtime) {
    auto result = popShared(runtime);
    std::lock_guard<std::mutex> lock(mutex_);
    current_[runtime] = result;
    return result.get();
  }
 private:
  std::map<Runtime *, Packet> current_;
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
  virtual base::Status synchronize();

  base::Status setWorkers(int worker_num,
                          std::vector<base::DeviceType> device_types =
                              std::vector<base::DeviceType>());

  virtual base::Status init(
      std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository,
      std::vector<device::Tensor *> &input_tensors,
      std::vector<device::Tensor *> &output_tensors, bool is_dynamic_shape,
      base::ShapeMap max_shape,
      TensorPoolType tensor_pool_type =
          kTensorPool1DSharedObjectTypeGreedyBySizeImprove,
      bool is_external_tensor_pool_memory = false) = 0;
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

  /**
   * @brief 将输入tensor复制到输入tensor
   *
   * @param tensor
   * @return base::Status
   */
  virtual base::Status copyToInputTensor(device::Tensor *tensor) = 0;

  /**
   * @brief 获取推理后的输出tensor
   *
   * @param name
   * @param device_type
   * @param is_copy
   * @param data_format
   * @return device::Tensor*
   */
  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format) = 0;

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
  bool is_external_tensor_pool_memory_ = false;
  TensorPool *tensor_pool_ = nullptr;
  bool is_dynamic_shape_ = false;                // 是否是动态shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  bool is_pure_dynamic_shape_ = false;
  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;
  std::vector<device::Tensor *> input_tensors_;
  std::vector<device::Tensor *> output_tensors_;
  int worker_num_ = 1;
  std::vector<base::DeviceType> device_types_;
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
