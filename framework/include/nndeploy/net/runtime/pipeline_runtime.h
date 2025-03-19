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

class PipelineTensor {
 public:
  PipelineTensor() {};
  virtual ~PipelineTensor() {};
  TensorWrapper *tensor_;
  std::vector<device::Tensor *> tensors_;
  std::vector<SequentialRuntime *> producers_;
  std::vector<SequentialRuntime *> consumers_;
};

class PipelineRuntime : public Runtime {
 public:
  PipelineRuntime(const base::DeviceType &device_type);
  virtual ~PipelineRuntime();

  // 设置流水线并行数量
  base::Status setWorkers(int num, std::vector<base::DeviceType> device_types =
                                       std::vector<base::DeviceType>());
  virtual base::Status init(
      std::vector<TensorWrapper *> &tensor_repository,
      std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
      base::ShapeMap max_shape,
      TensorPoolType tensor_pool_type =
          kTensorPool1DSharedObjectTypeGreedyBySizeImprove);
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();

  void commitThreadPool();
  base::EdgeUpdateFlag updateInput();

 private:
  int pipeline_parallel_num_ = 1;
  std::vector<base::DeviceType> device_types_;
  std::vector<SequentialRuntime *> sequential_runtimes_;
  std::map<TensorWrapper *, PipelineTensor *> input_output_tensors_;
  thread_pool::ThreadPool *thread_pool_ = nullptr;
};

}  // namespace net
}  // namespace nndeploy

#endif /* D98B9E43_6DEF_4878_822C_D1EA706E79EF */
