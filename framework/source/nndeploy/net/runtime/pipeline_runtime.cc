

#include "nndeploy/net/runtime/pipeline_runtime.h"

#include "nndeploy/base/time_profiler.h"

namespace nndeploy {
namespace net {

TypeRuntimeRegister<TypeRuntimeCreator<PipelineRuntime>>
    g_pipeline_runtime_register_seq(base::ParallelType::kParallelTypePipeline);

PipelineRuntime::PipelineRuntime(const base::DeviceType &device_type)
    : Runtime(device_type) {};
PipelineRuntime::~PipelineRuntime() {};

base::Status PipelineRuntime::setWorkers(
    int num, std::vector<base::DeviceType> device_types) {
  base::Status status = base::kStatusCodeOk;
  pipeline_parallel_num_ = num;
  device_types_ = device_types;
  if (device_types_.empty()) {
    device_types_.resize(num, device_type_);
  }
  if (device_types_.size() != num) {
    NNDEPLOY_LOGE("device_types_.size() != num\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return status;
}

base::Status PipelineRuntime::init(
    std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
    base::ShapeMap max_shape, TensorPoolType tensor_pool_type) {
  base::Status status = base::kStatusCodeOk;

  // 根据pipeline_parallel_num_切分模型
  if (pipeline_parallel_num_ <= 0) {
    NNDEPLOY_LOGE("pipeline_parallel_num_ <= 0\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 平均切分op_repository
  int ops_per_stage = op_repository.size() / pipeline_parallel_num_;
  if (ops_per_stage == 0) {
    ops_per_stage = 1;
  }

  // 为每个阶段创建SequentialRuntime
  for (int i = 0; i < pipeline_parallel_num_; ++i) {
    // 计算当前阶段的op范围
    int start_idx = i * ops_per_stage;
    int end_idx = (i == pipeline_parallel_num_ - 1) ? op_repository.size()
                                                    : (i + 1) * ops_per_stage;

    if (start_idx >= op_repository.size()) {
      break;  // 如果没有足够的op，提前结束
    }

    // 为当前阶段创建op和tensor子集
    std::vector<OpWrapper *> stage_ops;
    std::vector<TensorWrapper *> stage_tensors;

    // 添加当前阶段的ops
    for (int j = start_idx; j < end_idx; ++j) {
      stage_ops.push_back(op_repository[j]);
      for (auto tensor : tensor_repository) {
        if (std::find(tensor->producers_.begin(), tensor->producers_.end(),
                      op_repository[j]) != tensor->producers_.end()) {
          insertUnique(stage_tensors, tensor);
        }
        if (std::find(tensor->consumers_.begin(), tensor->consumers_.end(),
                      op_repository[j]) != tensor->consumers_.end()) {
          insertUnique(stage_tensors, tensor);
        }
      }
    }

    // 创建SequentialRuntime
    SequentialRuntime *sequential_runtime =
        new SequentialRuntime(device_types_[i]);
    // 收集输入输出张量
    for (auto tensor : stage_tensors) {
      bool is_input = false;
      bool is_output = false;

      // 找输入tensor，生产者为空 or 生产者不在这个stage_ops种
      if (tensor->producers_.empty()) {
        is_input = true;
      } else {
        for (auto producer : tensor->producers_) {
          if (std::find(stage_ops.begin(), stage_ops.end(), producer) ==
              stage_ops.end()) {
            is_input = true;
          }
        }
      }
      // 找输入tensor，消费者为空 or 消费者不在这个stage_ops种
      if (tensor->consumers_.empty()) {
        is_output = true;
      } else {
        for (auto consumer : tensor->consumers_) {
          if (std::find(stage_ops.begin(), stage_ops.end(), consumer) ==
              stage_ops.end()) {
            is_output = true;
          }
        }
      }

      // 添加到相应列表
      if (is_input) {
        if (input_output_tensors_.find(tensor) == input_output_tensors_.end()) {
          input_output_tensors_[tensor] = new PipelineTensor();
          input_output_tensors_[tensor]->tensor_ = tensor;
        }
        insertUnique(input_output_tensors_[tensor]->producers_,
                     sequential_runtime);
      }
      if (is_output) {
        if (input_output_tensors_.find(tensor) == input_output_tensors_.end()) {
          input_output_tensors_[tensor] = new PipelineTensor();
          input_output_tensors_[tensor]->tensor_ = tensor;
        }
        insertUnique(input_output_tensors_[tensor]->consumers_,
                     sequential_runtime);
      }
    }
    sequential_runtime->setAllocateInputOutputTensor(false);
    status =
        sequential_runtime->init(stage_tensors, stage_ops, is_dynamic_shape,
                                 max_shape, tensor_pool_type);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("SequentialRuntime init failed at stage %d\n", i);
      return status;
    }

    // 存储创建的runtime
    sequential_runtimes_.push_back(sequential_runtime);
  }

  thread_pool_ = new thread_pool::ThreadPool(pipeline_parallel_num_);
  status = thread_pool_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "thread_pool_ init failed");
  // 将所有节点塞入线程池
  this->commitThreadPool();

  return status;
}
base::Status PipelineRuntime::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (int i = 0; i < sequential_runtimes_.size(); ++i) {
    status = sequential_runtimes_[i]->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("sequential_runtime %d deinit failed\n", i);
      return status;
    }
  }
  return status;
}

// 可以性能优化
base::Status PipelineRuntime::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_LOGI("PipelineRuntime not support reshape\n");
  return status;
}

base::Status PipelineRuntime::preRun() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status PipelineRuntime::run() {
  base::Status status = base::kStatusCodeOk;
  return status;
}
base::Status PipelineRuntime::postRun() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

void PipelineRuntime::commitThreadPool() {
  // NNDEPLOY_LOGE("ppe run Thread ID: %d.\n", std::this_thread::get_id());
  for (auto iter : sequential_runtimes_) {
    auto func = [iter, this]() -> base::Status {
      base::Status status = base::kStatusCodeOk;
      while (true) {
        base::EdgeUpdateFlag edge_update_flag = this->updateInput();
        if (edge_update_flag == base::kEdgeUpdateFlagComplete) {
          status = iter->preRun();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "sequentialRuntime preRun failed!\n");
          status = iter->run();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "sequentialRuntime run failed!\n");
          status = iter->postRun();
          NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                                 "sequentialRuntime postRun failed!\n");
        } else if (edge_update_flag == base::kEdgeUpdateFlagTerminate) {
          break;
        } else {
          NNDEPLOY_LOGE("Failed to sequentialRuntime[%p] updateInput();\n",
                        iter);
          status = base::kStatusCodeErrorDag;
          break;
        }
      }
      return status;
    };
    thread_pool_->commit(std::bind(func));
  }
}

base::EdgeUpdateFlag PipelineRuntime::updateInput() {
  base::EdgeUpdateFlag edge_update_flag = base::kEdgeUpdateFlagComplete;
  for (auto iter : sequential_runtimes_) {
    // edge_update_flag = iter->updateInput();
    if (edge_update_flag != base::kEdgeUpdateFlagComplete) {
      break;
    }
  }
  return edge_update_flag;
}

}  // namespace net
}  // namespace nndeploy