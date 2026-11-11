#ifndef _NNDEPLOY_NET_RUNTIME_PIPELINE_RUNTIME_H_
#define _NNDEPLOY_NET_RUNTIME_PIPELINE_RUNTIME_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/device/device.h"
#include "nndeploy/net/runtime.h"
#include "nndeploy/net/runtime/sequential_runtime.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/util.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace net {

/**
 * @brief 流水线运行时阶段结构体，表示流水线中的一个执行阶段
 *
 * 流水线运行时将整个计算图分解为多个顺序执行的阶段，每个阶段包含一组Op和张量。
 * 这些阶段可以并行执行以提高吞吐量。
 */
struct PipelineRuntimeStage {
  std::vector<OpWrapper *> op_wrappers_;
  std::vector<TensorWrapper *> tensor_wrappers_;
  std::vector<device::Tensor *> input_tensors_, output_tensors_;
  std::vector<std::unique_ptr<TensorWrapper>> owned_wrappers_;
  std::vector<std::unique_ptr<device::Tensor>> owned_tensors_;
  std::map<device::Tensor *, device::Tensor *> tensor_map_;
  std::vector<std::pair<PipelineTensor *, device::Tensor *>> inputs_, outputs_;
};

class PipelineRuntime : public Runtime {
 public:
  explicit PipelineRuntime(const base::DeviceType &device_type);
  ~PipelineRuntime() override;
  base::Status setWorkers(int worker_num, std::vector<base::DeviceType> device_types = {});
  base::Status init(std::vector<TensorWrapper *> &tensor_repository,
                    std::vector<OpWrapper *> &op_repository,
                    std::vector<device::Tensor *> &input_tensors,
                    std::vector<device::Tensor *> &output_tensors,
                    bool is_dynamic_shape, base::ShapeMap max_shape,
                    TensorPoolType tensor_pool_type = kTensorPool1DSharedObjectTypeGreedyBySizeImprove,
                    bool is_external_tensor_pool_memory = false) override;
  base::Status deinit() override;
  base::Status reshape(base::ShapeMap &shape_map) override;
  base::Status preRun() override;
  base::Status run() override;
  base::Status postRun() override;
  base::Status synchronize() override;
  int64_t getMemorySize() override { return -1; }
  base::Status setMemory(device::Buffer *) override { return base::kStatusCodeErrorNotSupport; }
  void commitThreadPool();
  base::Status copyToInputTensor(device::Tensor *tensor) override;
  device::Tensor *getOutputTensorAfterRun(const std::string &name,
      base::DeviceType device_type, bool is_copy, base::DataFormat data_format) override;

 private:
  void stop(base::Status status);
  std::vector<SequentialRuntime *> sequential_runtimes_;
  std::vector<std::unique_ptr<PipelineRuntimeStage>> stages_;
  std::map<device::Tensor *, std::unique_ptr<PipelineTensor>> queues_;
  std::map<device::Tensor *, PipelineTensor::Packet> pending_inputs_;
  std::map<std::string, PipelineTensor::Packet> borrowed_outputs_;
  thread_pool::ThreadPool *thread_pool_ = nullptr;
  std::mutex pipeline_mutex_;
  std::condition_variable pipeline_cv_;
  size_t run_size_ = 0;
  std::vector<size_t> completed_;
  base::Status status_ = base::kStatusCodeOk;
  bool accepting_ = false, stopping_ = false;
};

} // namespace net
} // namespace nndeploy
#endif
