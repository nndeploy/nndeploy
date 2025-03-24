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

base::Status PipelineRuntime::setInputOutputTensors(
    std::map<device::Tensor *, PipelineTensor *> &input_output_tensors) {
  input_output_tensors_ = input_output_tensors;
  for (auto &item : input_output_tensors_) {
    input_output_set_.insert(item.first);
  }
  return base::kStatusCodeOk;
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

  if (device_types_.empty()) {
    device_types_.resize(pipeline_parallel_num_, device_type_);
    for (int i = 1; i < pipeline_parallel_num_; ++i) {
      device_types_[i].device_id_ = device_types_[0].device_id_ + i;
    }
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

    NNDEPLOY_LOGI("stage_ops.size() %d\n", stage_ops.size());
    NNDEPLOY_LOGI("stage_tensors.size() %d\n", stage_tensors.size());

    // 创建SequentialRuntime
    SequentialRuntime *sequential_runtime =
        new SequentialRuntime(device_types_[i]);

    // 收集输入输出张量
    for (auto tensor : stage_tensors) {
      bool is_input = false;
      bool is_output = false;

      if (tensor->is_weight_) {
        continue;
      }
      if (tensor->name_.empty()) {
        continue;
      }

      NNDEPLOY_LOGI("tensor %s\n", tensor->name_.c_str());

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
        NNDEPLOY_LOGI("tensor %s is input\n", tensor->name_.c_str());
        if (input_output_tensors_.find(tensor->tensor_) ==
            input_output_tensors_.end()) {
          input_output_tensors_[tensor->tensor_] = new PipelineTensor();
        }
        insertUnique(input_output_tensors_[tensor->tensor_]->consumers_,
                     (Runtime *)sequential_runtime);
        input_output_tensors_[tensor->tensor_]
            ->current_index_[sequential_runtime] = 0;
      }
      if (is_output) {
        NNDEPLOY_LOGI("tensor %s is output\n", tensor->name_.c_str());
        if (input_output_tensors_.find(tensor->tensor_) ==
            input_output_tensors_.end()) {
          input_output_tensors_[tensor->tensor_] = new PipelineTensor();
        }
        insertUnique(input_output_tensors_[tensor->tensor_]->producers_,
                     (Runtime *)sequential_runtime);
      }
    }

    sequential_runtime->setAllocateInputOutputTensor(true);

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

  // NNDEPLOY_LOGI("sequential_runtimes_.size() %d\n",
  //               sequential_runtimes_.size());
  // // 重置流水线状态
  // resetPipeline();

  thread_pool_ = new thread_pool::ThreadPool(pipeline_parallel_num_);
  status = thread_pool_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "thread_pool_ init failed");

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

void PipelineRuntime::resetPipeline() {
  // std::lock_guard<std::mutex> lock(pipeline_mutex_);
  // pipeline_complete_ = false;
  // completed_stages_ = 0;
  // for (auto &pair : input_output_tensors_) {
  //   PipelineTensor *pipeline_tensor = pair.second;
  //   for (auto &index : pipeline_tensor->current_index_) {
  //     index.second = 0;
  //   }
  //   for (auto tensor : pipeline_tensor->tensors_) {
  //     if (input_output_set_.find(tensor) != input_output_set_.end()) {
  //       delete tensor;
  //     }
  //   }
  //   pipeline_tensor->tensors_.clear();
  // }
  return;
}

bool PipelineRuntime::isPipelineComplete() { return pipeline_complete_; }

void PipelineRuntime::commitThreadPool() {
  for (int i = 0; i < sequential_runtimes_.size(); ++i) {
    SequentialRuntime *runtime = sequential_runtimes_[i];
    auto func = [runtime, this, i]() -> base::Status {
      base::Status status = base::kStatusCodeOk;

      while (true) {
        for (auto &pair : input_output_tensors_) {
          PipelineTensor *pipeline_tensor = pair.second;

          // NNDEPLOY_LOGI("tensor name %s\n", pair.first->getName().c_str());

          // 检查这个张量是否是当前阶段的输入
          if (std::find(pipeline_tensor->consumers_.begin(),
                        pipeline_tensor->consumers_.end(),
                        runtime) != pipeline_tensor->consumers_.end()) {
            device::Tensor *tensor = pipeline_tensor->pop(runtime);
            if (tensor == nullptr) {
              break;
            }
            device::Tensor *real_tensor = pair.first;
            if (real_tensor->getDevice() == tensor->getDevice()) {
              real_tensor->justModify(tensor->getBuffer());
            } else {
              tensor->copyTo(real_tensor);
            }
            static int count = 0;
            if (count == 0) {
              std::string filename =
                  "pipeline_tensor" + tensor->getName() + ".csv";
              size_t pos = 0;
              while ((pos = filename.find('/')) != std::string::npos) {
                filename.replace(pos, 1, "_");
              }
              std::ofstream input_file(filename, std::ios::trunc);
              if (input_file.is_open()) {
                tensor->print(input_file);
                input_file.close();
              } else {
                NNDEPLOY_LOGE("can't open file: %s", filename.c_str());
              }
            }
            if (count == 0) {
              std::string filename =
                  "real_tensor" + real_tensor->getName() + ".csv";
              size_t pos = 0;
              while ((pos = filename.find('/')) != std::string::npos) {
                filename.replace(pos, 1, "_");
              }
              std::ofstream input_file(filename, std::ios::trunc);
              if (input_file.is_open()) {
                real_tensor->print(input_file);
                input_file.close();
              } else {
                NNDEPLOY_LOGE("can't open file: %s", filename.c_str());
              }
            }
            count++;
          }
        }

        // 执行当前阶段
        status = runtime->preRun();
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("sequentialRuntime preRun failed!\n");
          break;
        }
        status = runtime->run();
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("sequentialRuntime run failed!\n");
          break;
        }
        status = runtime->postRun();
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("sequentialRuntime postRun failed!\n");
          break;
        }
        status = stream_->synchronize();

        // 更新输出张量状态
        for (auto &pair : input_output_tensors_) {
          PipelineTensor *pipeline_tensor = pair.second;
          device::Tensor *output_tensor = pair.first;
          // 检查这个张量是否是当前阶段的输出
          if (std::find(pipeline_tensor->producers_.begin(),
                        pipeline_tensor->producers_.end(),
                        runtime) != pipeline_tensor->producers_.end()) {
            device::Tensor *tensor = output_tensor->clone();
            static int count = 0;
            if (count == 0) {
              std::string path = "./output_tensor";
              std::string filename = path + std::to_string(count) + ".csv";
              std::ofstream file_stream(filename.c_str());
              output_tensor->print(file_stream);
              file_stream.close();
            }
            count++;
            pipeline_tensor->push(tensor);
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