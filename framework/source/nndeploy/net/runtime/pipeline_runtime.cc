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
    int worker_num, std::vector<base::DeviceType> device_types) {
  base::Status status = base::kStatusCodeOk;
  worker_num_ = worker_num;
  device_types_ = device_types;
  return status;
}

base::Status PipelineRuntime::init(
    std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository,
    std::vector<device::Tensor *> &input_tensors,
    std::vector<device::Tensor *> &output_tensors, bool is_dynamic_shape,
    base::ShapeMap max_shape, TensorPoolType tensor_pool_type) {
  base::Status status = base::kStatusCodeOk;

  input_tensors_ = input_tensors;
  output_tensors_ = output_tensors;

  // 根据worker_num_切分模型
  if (worker_num_ <= 0) {
    NNDEPLOY_LOGE("worker_num_ <= 0\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (device_types_.empty()) {
    device_types_.resize(worker_num_, device_type_);
    for (int i = 1; i < worker_num_; ++i) {
      device_types_[i].device_id_ = device_types_[0].device_id_ + i;
    }
  }

  NNDEPLOY_LOGI("worker_num_ %d\n", worker_num_);

  // 平均切分op_repository
  int ops_per_stage = op_repository.size() / worker_num_;
  if (ops_per_stage == 0) {
    ops_per_stage = 1;
  }

  NNDEPLOY_LOGI("ops_per_stage %d\n", ops_per_stage);

  // 为每个阶段创建SequentialRuntime
  for (int i = 0; i < worker_num_; ++i) {
    // 计算当前阶段的op范围
    int start_idx = i * ops_per_stage;
    int end_idx =
        (i == worker_num_ - 1) ? op_repository.size() : (i + 1) * ops_per_stage;

    NNDEPLOY_LOGI("start_idx %d, end_idx %d\n", start_idx, end_idx);

    if (start_idx >= op_repository.size()) {
      break;  // 如果没有足够的op，提前结束
    }

    // 为当前阶段创建op和tensor子集
    std::shared_ptr<PipelineRuntimeStage> stage =
        std::make_shared<PipelineRuntimeStage>();
    // std::vector<OpWrapper *> stage->op_wrappers_;
    // std::vector<TensorWrapper *> stage->tensor_wrappers_;

    // 添加当前阶段的ops
    for (int j = start_idx; j < end_idx; ++j) {
      stage->op_wrappers_.push_back(op_repository[j]);
      for (auto tensor : tensor_repository) {
        if (std::find(tensor->producers_.begin(), tensor->producers_.end(),
                      op_repository[j]) != tensor->producers_.end()) {
          insertUnique(stage->tensor_wrappers_, tensor);
        }
        if (std::find(tensor->consumers_.begin(), tensor->consumers_.end(),
                      op_repository[j]) != tensor->consumers_.end()) {
          insertUnique(stage->tensor_wrappers_, tensor);
        }
      }
    }

    NNDEPLOY_LOGI("stage->op_wrappers_.size() %d\n",
                  stage->op_wrappers_.size());
    NNDEPLOY_LOGI("stage->tensor_wrappers_.size() %d\n",
                  stage->tensor_wrappers_.size());

    // 创建SequentialRuntime
    SequentialRuntime *sequential_runtime =
        new SequentialRuntime(device_types_[i]);

    // std::vector<device::Tensor *> stage->input_tensors_;
    // std::vector<device::Tensor *> stage->output_tensors_;

    // 收集输入输出张量
    for (auto tensor : stage->tensor_wrappers_) {
      bool is_input = false;
      bool is_output = false;

      if (tensor->is_weight_) {
        continue;
      }
      if (tensor->producers_.empty() && tensor->consumers_.empty()) {
        NNDEPLOY_LOGI("tensor %s is not used\n", tensor->name_.c_str());
        continue;
      }
      if (tensor->name_.empty()) {
        continue;
      }

      // 找输入tensor，生产者为空 or 生产者不在这个stage->op_wrappers_种
      if (tensor->producers_.empty()) {
        is_input = true;
      } else {
        bool is_producer_in_stage = false;
        for (auto producer : tensor->producers_) {
          if (std::find(stage->op_wrappers_.begin(), stage->op_wrappers_.end(),
                        producer) == stage->op_wrappers_.end()) {
            is_producer_in_stage = true;
          } else {
            is_producer_in_stage = false;
            break;
          }
        }
        if (is_producer_in_stage) {
          is_input = true;
          tensor->producers_.clear();
        }
      }
      // 找输入tensor，消费者为空 or 消费者不在这个stage->op_wrappers_种
      if (tensor->consumers_.empty()) {
        is_output = true;
      } else {
        bool is_consumer_in_stage = false;
        for (auto consumer : tensor->consumers_) {
          if (std::find(stage->op_wrappers_.begin(), stage->op_wrappers_.end(),
                        consumer) == stage->op_wrappers_.end()) {
            is_consumer_in_stage = true;
          } else {
            is_consumer_in_stage = false;
            break;
          }
        }
        if (is_consumer_in_stage) {
          is_output = true;
          tensor->consumers_.clear();
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
        insertUnique(stage->input_tensors_, tensor->tensor_);
      }
      if (is_output) {
        NNDEPLOY_LOGI("tensor %s is output\n", tensor->name_.c_str());
        if (input_output_tensors_.find(tensor->tensor_) ==
            input_output_tensors_.end()) {
          input_output_tensors_[tensor->tensor_] = new PipelineTensor();
        }
        insertUnique(input_output_tensors_[tensor->tensor_]->producers_,
                     (Runtime *)sequential_runtime);
        if (std::find(output_tensors.begin(), output_tensors.end(),
                      tensor->tensor_) == output_tensors.end()) {
          insertUnique(input_output_tensors_[tensor->tensor_]->consumers_,
                       (Runtime *)nullptr);
          input_output_tensors_[tensor->tensor_]->current_index_[nullptr] = 0;
        }
        insertUnique(stage->output_tensors_, tensor->tensor_);
      }
    }

    // // sequential_runtime交互的tensor，分为多个
    // // 1. 输入tensor
    // std::vector<device::Tensor *> new_stage->input_tensors_;
    // for (auto tensor : stage->input_tensors_) {
    //   device::Tensor *new_tensor =
    //       new device::Tensor(tensor->getDesc(), tensor->getName());
    //   insertUnique(new_stage->input_tensors_, new_tensor);
    //   for (auto op : stage->op_wrappers_) {
    //     if (op->op_->getInputTensor(tensor->getName()) != nullptr) {
    //       op->op_->replaceInputTensor(new_tensor->getName(), new_tensor);
    //     }
    //     if (op->op_->getOutputTensor(tensor->getName()) != nullptr) {
    //       op->op_->replaceOutputTensor(new_tensor->getName(), new_tensor);
    //     }
    //   }
    //   for (auto stage_tensor : stage->tensor_wrappers_) {
    //     if (stage_tensor->name_ == tensor->getName()) {
    //       stage_tensor->tensor_ = new_tensor;
    //       break;
    //     }
    //   }
    // }
    // // 2. 输出tensor
    // std::vector<device::Tensor *> new_stage->output_tensors_;
    // for (auto tensor : stage->output_tensors_) {
    //   device::Tensor *new_tensor =
    //       new device::Tensor(tensor->getDesc(), tensor->getName());
    //   insertUnique(new_stage->output_tensors_, new_tensor);
    //   for (auto op : stage->op_wrappers_) {
    //     if (op->op_->getInputTensor(tensor->getName()) != nullptr) {
    //       op->op_->replaceInputTensor(new_tensor->getName(), new_tensor);
    //     }
    //     if (op->op_->getOutputTensor(tensor->getName()) != nullptr) {
    //       op->op_->replaceOutputTensor(new_tensor->getName(), new_tensor);
    //     }
    //   }
    //   for (auto stage_tensor : stage->tensor_wrappers_) {
    //     if (stage_tensor->name_ == tensor->getName()) {
    //       stage_tensor->tensor_ = new_tensor;
    //       break;
    //     }
    //   }
    // }
    // status = sequential_runtime->init(
    //     stage->tensor_wrappers_, stage->op_wrappers_,
    //     new_stage->input_tensors_, new_stage->output_tensors_,
    //     is_dynamic_shape, max_shape, tensor_pool_type);
    // if (status != base::kStatusCodeOk) {
    //   NNDEPLOY_LOGE("SequentialRuntime init failed at stage %d\n", i);
    //   return status;
    // }
    sequential_runtime_stage_stages_[sequential_runtime] = stage;

    // 存储创建的runtime
    sequential_runtimes_.push_back(sequential_runtime);
  }

  for (int i = 0; i < sequential_runtimes_.size(); ++i) {
    SequentialRuntime *sequential_runtime = sequential_runtimes_[i];
    std::shared_ptr<PipelineRuntimeStage> stage =
        sequential_runtime_stage_stages_[sequential_runtime];
    // sequential_runtime交互的tensor，分为多个
    // 1. 输入tensor
    std::vector<device::Tensor *> new_stage_input_tensors_;
    for (auto tensor : stage->input_tensors_) {
      device::Tensor *new_tensor =
          new device::Tensor(tensor->getDesc(), tensor->getName());
      insertUnique(new_stage_input_tensors_, new_tensor);
      for (auto op : stage->op_wrappers_) {
        if (op->op_->getInputTensor(tensor->getName()) != nullptr) {
          op->op_->replaceInputTensor(new_tensor->getName(), new_tensor);
        }
        if (op->op_->getOutputTensor(tensor->getName()) != nullptr) {
          op->op_->replaceOutputTensor(new_tensor->getName(), new_tensor);
        }
      }
      for (auto stage_tensor : stage->tensor_wrappers_) {
        if (stage_tensor->tensor_->getName() == tensor->getName()) {
          stage_tensor->tensor_ = new_tensor;
          break;
        }
      }
    }
    // 2. 输出tensor
    std::vector<device::Tensor *> new_stage_output_tensors_;
    for (auto tensor : stage->output_tensors_) {
      device::Tensor *new_tensor =
          new device::Tensor(tensor->getDesc(), tensor->getName());
      insertUnique(new_stage_output_tensors_, new_tensor);
      for (auto op : stage->op_wrappers_) {
        if (op->op_->getInputTensor(tensor->getName()) != nullptr) {
          op->op_->replaceInputTensor(new_tensor->getName(), new_tensor);
        }
        if (op->op_->getOutputTensor(tensor->getName()) != nullptr) {
          op->op_->replaceOutputTensor(new_tensor->getName(), new_tensor);
        }
      }
      for (auto stage_tensor : stage->tensor_wrappers_) {
        if (stage_tensor->tensor_->getName() == tensor->getName()) {
          stage_tensor->tensor_ = new_tensor;
          break;
        }
      }
    }
    NNDEPLOY_LOGE("new_stage_input_tensors_.size() %d\n",
                  new_stage_input_tensors_.size());
    NNDEPLOY_LOGE("new_stage_output_tensors_.size() %d\n",
                  new_stage_output_tensors_.size());
    status = sequential_runtime->init(
        stage->tensor_wrappers_, stage->op_wrappers_, new_stage_input_tensors_,
        new_stage_output_tensors_, is_dynamic_shape, max_shape,
        tensor_pool_type);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("SequentialRuntime init failed at stage %d\n", i);
      return status;
    }
    stage->input_tensors_ = new_stage_input_tensors_;
    stage->output_tensors_ = new_stage_output_tensors_;
    NNDEPLOY_LOGI("stage->input_tensors_.size() %d\n",
                  stage->input_tensors_.size());
    NNDEPLOY_LOGI("stage->output_tensors_.size() %d\n",
                  stage->output_tensors_.size());
  }

  thread_pool_ = new thread_pool::ThreadPool(worker_num_);
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
  NNDEPLOY_LOGI("PipelineRuntime deinit\n");

  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  pipeline_cv_.wait(lock, [this]() {
    bool flag = completed_size_ == run_size_;
    return flag;
  });
  NNDEPLOY_LOGI("PipelineRuntime deinit\n");
  for (auto &pair : input_output_tensors_) {
    PipelineTensor *pipeline_tensor = pair.second;
    std::lock_guard<std::mutex> lock(pipeline_tensor->mutex_);
    pipeline_tensor->is_finish_ = true;
    pipeline_tensor->cv_.notify_all();
  }

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

base::Status PipelineRuntime::allocateInput() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PipelineRuntime::allocateOutput() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PipelineRuntime::deallocateInput() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PipelineRuntime::deallocateOutput() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PipelineRuntime::preRun() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PipelineRuntime::run() {
  base::Status status = base::kStatusCodeOk;
  run_size_++;
  return status;
}

base::Status PipelineRuntime::postRun() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

void PipelineRuntime::commitThreadPool() {
  for (int i = 0; i < sequential_runtimes_.size(); ++i) {
    SequentialRuntime *runtime = sequential_runtimes_[i];
    auto func = [runtime, this, i]() -> base::Status {
      base::Status status = base::kStatusCodeOk;
      device::Device *device = device::getDevice(device_types_[i]);
      device->bindThread();
      auto stage = sequential_runtime_stage_stages_[runtime];

      while (true) {
        bool is_finish = false;
        for (auto &pair : input_output_tensors_) {
          PipelineTensor *pipeline_tensor = pair.second;
          NNDEPLOY_LOGI("tensor name %s\n", pair.first->getName().c_str());
          if (std::find(pipeline_tensor->consumers_.begin(),
                        pipeline_tensor->consumers_.end(),
                        runtime) != pipeline_tensor->consumers_.end()) {
            device::Tensor *tensor = pipeline_tensor->pop(runtime);
            if (tensor == nullptr) {
              is_finish = true;
              break;
            }
            status = runtime->copyToInputTensor(tensor);
            if (status != base::kStatusCodeOk) {
              NNDEPLOY_LOGE("copyToInputTensor failed!\n");
              break;
            }
          }
        }
        if (is_finish) {
          break;
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
        std::string name = "synchronize_" + std::to_string((size_t)runtime);
        NNDEPLOY_TIME_POINT_START(name.c_str());
        NNDEPLOY_LOGI("name %s\n", name.c_str());
        runtime->synchronize();
        NNDEPLOY_LOGI("name %s\n", name.c_str());
        NNDEPLOY_TIME_POINT_END(name.c_str());

        // 更新输出张量状态
        // for (auto &pair : input_output_tensors_) {
        //   PipelineTensor *pipeline_tensor = pair.second;
        //   device::Tensor *output_tensor = pair.first;
        //   // 检查这个张量是否是当前阶段的输出
        //   if (std::find(pipeline_tensor->producers_.begin(),
        //                 pipeline_tensor->producers_.end(),
        //                 runtime) != pipeline_tensor->producers_.end()) {
        //     device::Tensor *tensor = output_tensor->clone();
        //     pipeline_tensor->push(tensor);
        //   }
        // }
        // 测试代码
        for (auto tensor : stage->input_tensors_) {
          NNDEPLOY_LOGI("%p, tensor name %s\n", runtime,
                        tensor->getName().c_str());
        }
        for (auto &pair : input_output_tensors_) {
          PipelineTensor *pipeline_tensor = pair.second;
          NNDEPLOY_LOGI("%p, tensor name %s\n", runtime,
                        pair.first->getName().c_str());
          // if (std::find(pipeline_tensor->consumers_.begin(),
          //               pipeline_tensor->consumers_.end(),
          //               runtime) != pipeline_tensor->consumers_.end()) {
          //   NNDEPLOY_LOGI("%p, tensor name %s\n", runtime,
          //                 pair.first->getName().c_str());
          // }
        }
        for (auto tensor : stage->output_tensors_) {
          for (auto &pair : input_output_tensors_) {
            PipelineTensor *pipeline_tensor = pair.second;
            if (tensor->getName() == pair.first->getName()) {
              NNDEPLOY_LOGI("%p, tensor name %s\n", runtime,
                            tensor->getName().c_str());
              pipeline_tensor->push(tensor);
            }
          }
        }
        NNDEPLOY_LOGI("runtime %p 阶段完成\n", runtime);

        // 检查是否所有阶段都已完成
        // {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        if (i == sequential_runtimes_.size() - 1) {
          completed_size_++;
        }
        if (completed_size_ == run_size_) {
          pipeline_cv_.notify_all();
        }
        // }
      }
      static int count = 0;
      count++;
      NNDEPLOY_LOGI("%p 线程执行完成 %d\n", runtime, count);
      return status;
    };

    thread_pool_->commit(std::bind(func));
  }
}

base::Status PipelineRuntime::copyToInputTensor(device::Tensor *tensor) {
  device::Tensor *src_tensor = tensor;
  device::Tensor *dst_tensor = nullptr;
  for (auto input : input_tensors_) {
    if (input->getName() == src_tensor->getName()) {
      dst_tensor = input;
      break;
    }
  }
  if (dst_tensor == nullptr) {
    NNDEPLOY_LOGE("copyToInputTensor failed! input tensor not found!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  auto iter = input_output_tensors_.find(dst_tensor);
  if (iter == input_output_tensors_.end()) {
    NNDEPLOY_LOGE("input_output_tensors_ not found dst_tensor!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  PipelineTensor *pipeline_tensor = iter->second;
  device::Tensor *vec_dst_tensor = new device::Tensor(*src_tensor);
  static int count = 0;
  if (count == 0) {
    std::string filename = "vec_dst_tensor" + src_tensor->getName() + ".csv";
    size_t pos = 0;
    while ((pos = filename.find('/')) != std::string::npos) {
      filename.replace(pos, 1, "_");
    }
    std::ofstream input_file(filename, std::ios::trunc);
    if (input_file.is_open()) {
      vec_dst_tensor->print(input_file);
      input_file.close();
    } else {
      NNDEPLOY_LOGE("can't open file: %s", filename.c_str());
    }
  }
  count++;
  pipeline_tensor->push(vec_dst_tensor);
  return base::kStatusCodeOk;
}
device::Tensor *PipelineRuntime::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  device::Tensor *internal_output_tensor = nullptr;
  for (auto output : output_tensors_) {
    if (output->getName() == name) {
      internal_output_tensor = output;
      break;
    }
  }
  if (internal_output_tensor == nullptr) {
    NNDEPLOY_LOGE("Not exsit output[%s].\n", name.c_str());
    return nullptr;
  }
  auto iter = input_output_tensors_.find(internal_output_tensor);
  if (iter == input_output_tensors_.end()) {
    NNDEPLOY_LOGE("input_output_tensors_ not found internal_output_tensor!\n");
    return nullptr;
  }
  PipelineTensor *pipeline_internal_output_tensor = iter->second;
  NNDEPLOY_LOGI("pipeline_internal_output_tensor->tensors_.size() %d\n",
                pipeline_internal_output_tensor->tensors_.size());
  device::Tensor *pipeline_output_tensor =
      pipeline_internal_output_tensor->pop(nullptr);
  if (pipeline_output_tensor == nullptr) {
    NNDEPLOY_LOGE("pipeline_output_tensor is nullptr!\n");
    return nullptr;
  }
  NNDEPLOY_LOGI("pipeline_output_tensor NAME %s\n",
                pipeline_output_tensor->getName().c_str());
  bool flag = is_copy || (pipeline_output_tensor->getDevice() != device);
  device::Tensor *output_tensor = nullptr;
  if (flag) {
    output_tensor =
        new device::Tensor(device, pipeline_output_tensor->getDesc(), name);
    pipeline_output_tensor->copyTo(output_tensor);
    return output_tensor;
  } else {
    return pipeline_output_tensor;
  }
}

}  // namespace net
}  // namespace nndeploy