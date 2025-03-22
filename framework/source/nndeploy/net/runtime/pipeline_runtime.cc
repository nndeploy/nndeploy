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
  return status;
}

base::Status PipelineRuntime::init(
    std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
    base::ShapeMap max_shape, TensorPoolType tensor_pool_type) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_LOGI("PipelineRuntime init\n");

  // 根据pipeline_parallel_num_切分模型
  if (pipeline_parallel_num_ <= 0) {
    NNDEPLOY_LOGE("pipeline_parallel_num_ <= 0\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (device_types_.empty()) {
    device_types_.resize(pipeline_parallel_num_, device_type_);
  }
  for (int i = 1; i < pipeline_parallel_num_; ++i) {
    device_types_[i].device_id_ = device_types_[0].device_id_ + i;
  }

  NNDEPLOY_LOGI("pipeline_parallel_num_ %d\n", pipeline_parallel_num_);

  // 平均切分op_repository
  int ops_per_stage = op_repository.size() / pipeline_parallel_num_;
  if (ops_per_stage == 0) {
    ops_per_stage = 1;
  }

  NNDEPLOY_LOGI("ops_per_stage %d\n", ops_per_stage);

  // 为每个阶段创建SequentialRuntime
  for (int i = 0; i < pipeline_parallel_num_; ++i) {
    // 计算当前阶段的op范围
    int start_idx = i * ops_per_stage;
    int end_idx = (i == pipeline_parallel_num_ - 1) ? op_repository.size()
                                                    : (i + 1) * ops_per_stage;

    NNDEPLOY_LOGI("start_idx %d, end_idx %d\n", start_idx, end_idx);

    if (start_idx >= op_repository.size()) {
      break;  // 如果没有足够的op，提前结束
    }

    NNDEPLOY_LOGI("start_idx %d, end_idx %d\n", start_idx, end_idx);

    // 为当前阶段创建op和tensor子集
    std::vector<OpWrapper *> stage_ops;
    std::vector<TensorWrapper *> stage_tensors;

    NNDEPLOY_LOGI("stage_ops.size() %d\n", stage_ops.size());
    NNDEPLOY_LOGI("stage_tensors.size() %d\n", stage_tensors.size());

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

    NNDEPLOY_LOGI("stage_ops.size() %d\n", stage_ops.size());
    NNDEPLOY_LOGI("stage_tensors.size() %d\n", stage_tensors.size());

    // 创建SequentialRuntime
    SequentialRuntime *sequential_runtime =
        new SequentialRuntime(device_types_[i]);
    NNDEPLOY_LOGI("sequential_runtime %p\n", sequential_runtime);
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
        input_output_tensors_[tensor]->current_index_[sequential_runtime] = 0;
      }
    }
    NNDEPLOY_LOGI("input_output_tensors_.size() %d\n",
                  input_output_tensors_.size());
    sequential_runtime->setAllocateInputOutputTensor(true);
    NNDEPLOY_LOGI("sequential_runtime->setAllocateInputOutputTensor(true)");
    status =
        sequential_runtime->init(stage_tensors, stage_ops, is_dynamic_shape,
                                 max_shape, tensor_pool_type);
    NNDEPLOY_LOGI("sequential_runtime->init success");
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("SequentialRuntime init failed at stage %d\n", i);
      return status;
    }

    // 存储创建的runtime
    sequential_runtimes_.push_back(sequential_runtime);
  }

  NNDEPLOY_LOGI("sequential_runtimes_.size() %d\n",
                sequential_runtimes_.size());
  // 重置流水线状态
  resetPipeline();
  NNDEPLOY_LOGI("resetPipeline success");

  thread_pool_ = new thread_pool::ThreadPool(pipeline_parallel_num_);
  status = thread_pool_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "thread_pool_ init failed");

  NNDEPLOY_LOGI("PipelineRuntime init success\n");

  // 将所有节点塞入线程池
  this->commitThreadPool();

  NNDEPLOY_LOGI("PipelineRuntime commitThreadPool success\n");

  return status;
}
base::Status PipelineRuntime::deinit() {
  base::Status status = base::kStatusCodeOk;

  // 停止线程池
  if (thread_pool_ != nullptr) {
    thread_pool_->destroy();
    delete thread_pool_;
    thread_pool_ = nullptr;
  }

  // 释放 PipelineTensor 资源
  for (auto &pair : input_output_tensors_) {
    if (pair.second != nullptr) {
      delete pair.second;
      pair.second = nullptr;
    }
  }
  input_output_tensors_.clear();

  // 释放 SequentialRuntime 资源
  for (int i = 0; i < sequential_runtimes_.size(); ++i) {
    status = sequential_runtimes_[i]->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("sequential_runtime %d deinit failed\n", i);
      return status;
    }
    delete sequential_runtimes_[i];
  }
  sequential_runtimes_.clear();

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
  // 初始化输入张量的状态，只执行一次
  static bool initialized = false;
  if (!initialized) {
    for (auto &pair : input_output_tensors_) {
      PipelineTensor *pipeline_tensor = pair.second;
      if (pipeline_tensor->producers_.empty()) {
        // 这是网络的输入张量，标记为就绪
        std::lock_guard<std::mutex> lock(pipeline_tensor->mutex_);
        pipeline_tensor->is_ready_ = true;
        for (auto producer : pipeline_tensor->producers_) {
          pipeline_tensor->current_index_[producer] = 0;
        }
        pipeline_tensor->cv_.notify_all();
      }
    }
    initialized = true;
    NNDEPLOY_LOGI("PipelineRuntime preRun success\n");
  }
  return status;
}

base::Status PipelineRuntime::run() {
  base::Status status = base::kStatusCodeOk;
  // 等待流水线完成
  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [this]() { return this->isPipelineComplete(); });
  return status;
}

base::Status PipelineRuntime::postRun() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

void PipelineRuntime::resetPipeline() {
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  pipeline_complete_ = false;
  completed_stages_ = 0;
  pipeline_batch_size_ = 1;
  for (auto &pair : input_output_tensors_) {
    PipelineTensor *pipeline_tensor = pair.second;
    pipeline_tensor->is_ready_ = false;
    for (auto &index : pipeline_tensor->current_index_) {
      index.second = 0;
    }
    for (auto tensor : pipeline_tensor->tensors_) {
      delete tensor;
    }
    pipeline_tensor->tensors_.clear();
  }
}

bool PipelineRuntime::isPipelineComplete() { return pipeline_complete_; }

void PipelineRuntime::commitThreadPool() {
  for (int i = 0; i < sequential_runtimes_.size(); ++i) {
    SequentialRuntime *runtime = sequential_runtimes_[i];
    auto func = [runtime, this, i]() -> base::Status {
      base::Status status = base::kStatusCodeOk;

      while (true) {
        // 等待输入就绪
        bool inputs_ready = true;
        for (auto &pair : input_output_tensors_) {
          PipelineTensor *pipeline_tensor = pair.second;

          // 检查这个张量是否是当前阶段的输入
          if (std::find(pipeline_tensor->consumers_.begin(),
                        pipeline_tensor->consumers_.end(),
                        runtime) != pipeline_tensor->consumers_.end()) {
            // 等待输入就绪
            std::unique_lock<std::mutex> lock(pipeline_tensor->mutex_);
            pipeline_tensor->cv_.wait(lock, [pipeline_tensor]() {
              return pipeline_tensor->is_ready_;
            });

            // 将张量数据复制到当前阶段的输入
            if (pipeline_tensor->tensors_.size() >
                pipeline_tensor->current_index_[runtime]) {
              // 这里需要实现张量数据的复制逻辑
              // 从pipeline_tensor->tensors_[pipeline_tensor->current_index_]
              // 复制到runtime的输入
              // pipeline_tensor
              //     ->tensors_[pipeline_tensor->current_index_[runtime]]
              //     ->copyTo(pipeline_tensor->tensor_->tensor_);
            }
          }
        }

        // 执行当前阶段
        NNDEPLOY_LOGI("runtime->preRun\n");
        status = runtime->preRun();
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("sequentialRuntime preRun failed!\n");
          break;
        }
        NNDEPLOY_LOGI("runtime->run\n");
        status = runtime->run();
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("sequentialRuntime run failed!\n");
          break;
        }
        NNDEPLOY_LOGI("runtime->postRun\n");
        status = runtime->postRun();
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("sequentialRuntime postRun failed!\n");
          break;
        }
        NNDEPLOY_LOGI("runtime->updateOutputTensor\n");
        // 更新输出张量状态
        for (auto &pair : input_output_tensors_) {
          PipelineTensor *pipeline_tensor = pair.second;

          // 检查这个张量是否是当前阶段的输出
          if (std::find(pipeline_tensor->producers_.begin(),
                        pipeline_tensor->producers_.end(),
                        runtime) != pipeline_tensor->producers_.end()) {
            std::lock_guard<std::mutex> lock(pipeline_tensor->mutex_);
            // 将当前阶段的输出复制到pipeline_tensor
            // 这里需要实现张量数据的复制逻辑
            device::Tensor *tensor = pipeline_tensor->tensor_->tensor_->clone();
            pipeline_tensor->tensors_.push_back(tensor);

            pipeline_tensor->is_ready_ = true;
            pipeline_tensor->current_index_[runtime] =
                (pipeline_tensor->current_index_[runtime] + 1) %
                std::max(1, (int)pipeline_tensor->tensors_.size());
            pipeline_tensor->cv_.notify_all();
          }
        }

        // 检查是否是最后一个阶段，如果是则增加完成计数
        if (i == sequential_runtimes_.size() - 1) {
          std::lock_guard<std::mutex> lock(pipeline_mutex_);
          completed_stages_++;
          if (completed_stages_ >=
              pipeline_batch_size_) {  // 可以根据需要调整完成的批次数
            pipeline_complete_ = true;
            pipeline_cv_.notify_all();
          }
        }

        // 如果流水线已完成，退出循环
        {
          std::lock_guard<std::mutex> lock(pipeline_mutex_);
          if (pipeline_complete_) {
            break;
          }
        }
      }

      return status;
    };

    thread_pool_->commit(std::bind(func));
  }
}

}  // namespace net
}  // namespace nndeploy