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
          kTensorPool1DSharedObjectTypeGreedyBySizeImprove);
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status allocateInput();
  virtual base::Status allocateOutput();
  virtual base::Status deallocateInput();
  virtual base::Status deallocateOutput();

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
  std::vector<SequentialRuntime *> sequential_runtimes_;

  std::map<device::Tensor *, PipelineTensor *> input_output_tensors_;

  thread_pool::ThreadPool *thread_pool_ = nullptr;

  // 添加互斥锁和条件变量，用于同步流水线状态
  std::mutex pipeline_mutex_;
  bool pipeline_complete_ = false;
  int completed_stages_ = 0;
  int pipeline_batch_size_ = INT_MAX;
};

}  // namespace net
}  // namespace nndeploy

#endif /* D98B9E43_6DEF_4878_822C_D1EA706E79EF */
