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
  std::vector<OpWrapper *> op_wrappers_;  ///< 该阶段包含的Op包装器列表
  std::vector<TensorWrapper *>
      tensor_wrappers_;                          ///< 该阶段使用的张量包装器列表
  std::vector<device::Tensor *> input_tensors_;  ///< 该阶段的输入张量列表
  std::vector<device::Tensor *> output_tensors_;  ///< 该阶段的输出张量列表
};

class PipelineRuntime : public Runtime {
 public:
  PipelineRuntime(const base::DeviceType &device_type);
  virtual ~PipelineRuntime();

  // 设置流水线并行数量
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
      bool is_external_tensor_pool_memory = false);
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();

  void commitThreadPool();

  virtual base::Status copyToInputTensor(device::Tensor *tensor);
  virtual device::Tensor *getOutputTensorAfterRun(const std::string &name,
                                                  base::DeviceType device_type,
                                                  bool is_copy,
                                                  base::DataFormat data_format);

 private:
  /**
   * @brief 顺序运行时实例列表，每个实例负责流水线中的一个阶段
   *
   * 流水线运行时将计算任务分解为多个顺序执行的阶段，每个阶段由一个SequentialRuntime实例管理
   */
  std::vector<SequentialRuntime *> sequential_runtimes_;

  /**
   * @brief 顺序运行时与对应流水线阶段的映射关系
   *
   * 将每个SequentialRuntime实例映射到其负责的PipelineRuntimeStage，
   * 包含该阶段的所有操作和张量信息
   */
  std::map<SequentialRuntime *, std::shared_ptr<PipelineRuntimeStage>>
      sequential_runtime_stage_stages_;

  /**
   * @brief 输入输出张量映射表
   *
   * 将原始设备张量映射到流水线专用张量，用于管理跨阶段的数据传输和同步
   */
  std::map<device::Tensor *, PipelineTensor *> input_output_tensors_;

  /**
   * @brief 线程池实例，用于并行执行各流水线阶段
   *
   * 管理工作线程，实现流水线各阶段的并行调度执行
   */
  thread_pool::ThreadPool *thread_pool_ = nullptr;

  /**
   * @brief 流水线同步互斥锁
   *
   * 保护流水线共享状态，确保线程安全的状态更新
   */
  std::mutex pipeline_mutex_;

  /**
   * @brief 流水线同步条件变量
   *
   * 用于线程间的同步通知，协调各阶段的执行顺序和依赖关系
   */
  std::condition_variable pipeline_cv_;

  /**
   * @brief 当前提交到流水线的任务数量
   *
   * 记录已提交但尚未完全处理完的任务总数
   */
  int run_size_ = 0;

  /**
   * @brief 已完成处理的任务数量
   *
   * 记录已经完成处理的任务数，用于跟踪流水线进度和同步
   */
  int completed_size_ = 0;
};

}  // namespace net
}  // namespace nndeploy

#endif /* D98B9E43_6DEF_4878_822C_D1EA706E79EF */
