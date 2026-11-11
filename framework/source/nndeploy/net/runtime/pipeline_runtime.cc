#include "nndeploy/net/runtime/pipeline_runtime.h"

namespace nndeploy {
namespace net {
TypeRuntimeRegister<TypeRuntimeCreator<PipelineRuntime>>
    g_pipeline_runtime_register_seq(base::kParallelTypePipeline);

namespace {
base::Status snapshot(device::Tensor *source, PipelineTensor::Packet &result) {
  if (!source || !source->getBuffer()) return base::kStatusCodeErrorInvalidParam;
  auto copy = std::make_shared<device::Tensor>(source->getDevice(), source->getDesc(), source->getName());
  auto status = source->copyTo(copy.get());
  if (status == base::kStatusCodeOk) result = std::move(copy);
  return status;
}
}

PipelineRuntime::PipelineRuntime(const base::DeviceType &device_type) : Runtime(device_type) {}
PipelineRuntime::~PipelineRuntime() { if (!stages_.empty()) { stop(base::kStatusCodeNodeInterrupt); deinit(); } }
base::Status PipelineRuntime::setWorkers(int count, std::vector<base::DeviceType> devices) {
  return Runtime::setWorkers(count, std::move(devices));
}

base::Status PipelineRuntime::init(std::vector<TensorWrapper *> &tensors,
    std::vector<OpWrapper *> &ops, std::vector<device::Tensor *> &inputs,
    std::vector<device::Tensor *> &outputs, bool dynamic, base::ShapeMap max_shape,
    TensorPoolType pool_type, bool external_memory) {
  if (!stages_.empty() || worker_num_ <= 0 || ops.empty()) return base::kStatusCodeErrorInvalidParam;
  if (external_memory) return base::kStatusCodeErrorNotSupport;
  const size_t count = std::min<size_t>(worker_num_, ops.size());
  if (device_types_.empty()) device_types_.assign(count, device_type_);
  if (device_types_.size() < count) return base::kStatusCodeErrorInvalidParam;
  for (size_t i = 0; i < count; ++i)
    if (!device::getDevice(device_types_[i])) return base::kStatusCodeErrorInvalidParam;
  input_tensors_ = inputs;
  output_tensors_ = outputs;
  run_size_ = 0;
  completed_.assign(count, 0);
  status_ = base::kStatusCodeOk;
  stopping_ = false;
  auto queue = [&](device::Tensor *tensor) -> PipelineTensor * {
    auto &entry = queues_[tensor];
    if (!entry) entry.reset(new PipelineTensor());
    return entry.get();
  };
  // Keep the original graph's wrappers and Tensor pointers intact. Each stage
  // owns separate activation descriptors, storage, and tensor-pool metadata.
  for (size_t i = 0; i < count; ++i) {
    auto stage = std::unique_ptr<PipelineRuntimeStage>(new PipelineRuntimeStage());
    const size_t begin = i * ops.size() / count, end = (i + 1) * ops.size() / count;
    stage->op_wrappers_.assign(ops.begin() + begin, ops.begin() + end);
    auto contains = [&](OpWrapper *op) {
      return std::find(stage->op_wrappers_.begin(), stage->op_wrappers_.end(), op) != stage->op_wrappers_.end();
    };
    auto *runtime = new SequentialRuntime(device_types_[i]);
    sequential_runtimes_.push_back(runtime);
    for (auto *original : tensors) {
      bool used = std::any_of(original->producers_.begin(), original->producers_.end(), contains) ||
                  std::any_of(original->consumers_.begin(), original->consumers_.end(), contains);
      if (!used) continue;
      auto local = std::unique_ptr<device::Tensor>(original->is_weight_
          ? new device::Tensor(*original->tensor_)
          : new device::Tensor(original->tensor_->getDesc(), original->tensor_->getName()));
      auto wrapper = std::unique_ptr<TensorWrapper>(new TensorWrapper(*original));
      wrapper->tensor_ = local.get();
      auto *tensor = local.get();
      stage->tensor_map_[original->tensor_] = tensor;
      const bool produced_here = std::any_of(original->producers_.begin(), original->producers_.end(), contains);
      const bool is_input = !original->is_weight_ && !produced_here;
      const bool is_output = !original->is_weight_ && produced_here &&
          (original->consumers_.empty() ||
           std::any_of(original->consumers_.begin(), original->consumers_.end(), [&](OpWrapper *op) { return !contains(op); }) ||
           std::find(outputs.begin(), outputs.end(), original->tensor_) != outputs.end());
      if (is_input) {
        auto *q = queue(original->tensor_);
        q->consumers_.push_back(runtime); q->current_index_[runtime] = 0;
        stage->input_tensors_.push_back(tensor); stage->inputs_.push_back({q, tensor});
      }
      if (is_output) {
        auto *q = queue(original->tensor_);
        q->producers_.push_back(runtime);
        if (std::find(outputs.begin(), outputs.end(), original->tensor_) != outputs.end()) {
          q->consumers_.push_back(nullptr); q->current_index_[nullptr] = 0;
        }
        stage->output_tensors_.push_back(tensor); stage->outputs_.push_back({q, tensor});
      }
      wrapper->input_output_type_ = is_input ? kInput : is_output ? kOutput : kNone;
      for (auto *op : stage->op_wrappers_) {
        if (op->op_->getInputTensor(tensor->getName())) op->op_->replaceInputTensor(tensor->getName(), tensor);
        if (op->op_->getOutputTensor(tensor->getName())) op->op_->replaceOutputTensor(tensor->getName(), tensor);
      }
      stage->tensor_wrappers_.push_back(wrapper.get());
      stage->owned_wrappers_.push_back(std::move(wrapper));
      stage->owned_tensors_.push_back(std::move(local));
    }
    stages_.push_back(std::move(stage));
  }
  // Direct graph input -> graph output is also a valid boundary.
  for (auto *output : outputs) {
    auto *q = queue(output);
    if (!q->current_index_.count(nullptr)) {
      if (std::find(inputs.begin(), inputs.end(), output) == inputs.end()) {
        status_ = base::kStatusCodeErrorInvalidValue; deinit(); return status_;
      }
      q->consumers_.push_back(nullptr); q->current_index_[nullptr] = 0;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    auto &stage = *stages_[i];
    auto status = sequential_runtimes_[i]->init(stage.tensor_wrappers_, stage.op_wrappers_,
        stage.input_tensors_, stage.output_tensors_, dynamic, max_shape, pool_type, false);
    if (status != base::kStatusCodeOk) { status_ = status; deinit(); return status; }
  }
  thread_pool_ = new thread_pool::ThreadPool(count);
  auto status = thread_pool_->init();
  if (status != base::kStatusCodeOk) { status_ = status; deinit(); return status; }
  accepting_ = true;
  commitThreadPool();
  return base::kStatusCodeOk;
}

void PipelineRuntime::stop(base::Status status) {
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (status_ == base::kStatusCodeOk) status_ = status;
    stopping_ = true; accepting_ = false;
  }
  for (auto &entry : queues_) entry.second->setFinish();
  pipeline_cv_.notify_all();
}

base::Status PipelineRuntime::synchronize() {
  std::unique_lock<std::mutex> lock(pipeline_mutex_);
  auto target = run_size_;
  pipeline_cv_.wait(lock, [&] {
    return stopping_ || std::all_of(completed_.begin(), completed_.end(),
                                  [target](size_t done) { return done >= target; });
  });
  return status_;
}

base::Status PipelineRuntime::deinit() {
  {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    accepting_ = false;
  }
  if (thread_pool_) synchronize();
  stop(base::kStatusCodeOk);
  if (thread_pool_) {
    thread_pool_->destroy(); delete thread_pool_; thread_pool_ = nullptr;
  }
  for (size_t i = 0; i < sequential_runtimes_.size(); ++i) {
    auto status = sequential_runtimes_[i]->deinit();
    if (status_ == base::kStatusCodeOk) status_ = status;
    delete sequential_runtimes_[i];
    auto &stage = *stages_[i];
    for (auto &mapping : stage.tensor_map_) {
      for (auto *op : stage.op_wrappers_) {
        const auto &name = mapping.first->getName();
        if (op->op_->getInputTensor(name)) op->op_->replaceInputTensor(name, mapping.first);
        if (op->op_->getOutputTensor(name)) op->op_->replaceOutputTensor(name, mapping.first);
      }
    }
  }
  sequential_runtimes_.clear(); stages_.clear(); queues_.clear();
  pending_inputs_.clear(); borrowed_outputs_.clear();
  return status_;
}

base::Status PipelineRuntime::reshape(base::ShapeMap &) { return base::kStatusCodeErrorNotSupport; }
base::Status PipelineRuntime::preRun() { std::lock_guard<std::mutex> lock(pipeline_mutex_); return status_; }
base::Status PipelineRuntime::postRun() { return preRun(); }

base::Status PipelineRuntime::run() {
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (status_ != base::kStatusCodeOk) return status_;
  if (!accepting_) return base::kStatusCodeErrorInvalidValue;
  for (auto *input : input_tensors_)
    if (!pending_inputs_.count(input)) return base::kStatusCodeErrorInvalidParam;
  for (auto &entry : pending_inputs_) queues_.at(entry.first)->push(std::move(entry.second));
  pending_inputs_.clear();
  ++run_size_;
  pipeline_cv_.notify_all();
  return base::kStatusCodeOk;
}

void PipelineRuntime::commitThreadPool() {
  for (size_t i = 0; i < stages_.size(); ++i) {
    thread_pool_->commit([this, i] {
      auto *runtime = sequential_runtimes_[i];
      auto &stage = *stages_[i];
      device::getDevice(device_types_[i])->bindThread();
      while (true) {
        {
          std::unique_lock<std::mutex> lock(pipeline_mutex_);
          pipeline_cv_.wait(lock, [&] { return stopping_ || completed_[i] < run_size_; });
          if (stopping_) return;
        }
        base::Status status = base::kStatusCodeOk;
        try {
          for (auto &input : stage.inputs_) {
            auto frame = input.first->popShared(runtime);
            if (!frame) return;
            status = runtime->copyToInputTensor(frame.get());
            if (status != base::kStatusCodeOk) break;
          }
          if (status == base::kStatusCodeOk) status = runtime->preRun();
          if (status == base::kStatusCodeOk) status = runtime->run();
          if (status == base::kStatusCodeOk) status = runtime->postRun();
          if (status == base::kStatusCodeOk) status = runtime->synchronize();
          // Snapshot all outputs before publishing any of them.
          std::vector<PipelineTensor::Packet> frames;
          if (status == base::kStatusCodeOk) {
            for (auto &output : stage.outputs_) {
              PipelineTensor::Packet frame;
              status = snapshot(output.second, frame);
              if (status != base::kStatusCodeOk) break;
              frames.push_back(std::move(frame));
            }
          }
          if (status == base::kStatusCodeOk)
            for (size_t j = 0; j < frames.size(); ++j) stage.outputs_[j].first->push(std::move(frames[j]));
        } catch (...) { status = base::kStatusCodeErrorUnknown; }
        if (status != base::kStatusCodeOk) { stop(status); return; }
        {
          std::lock_guard<std::mutex> lock(pipeline_mutex_);
          ++completed_[i];
        }
        pipeline_cv_.notify_all();
      }
    });
  }
}

base::Status PipelineRuntime::copyToInputTensor(device::Tensor *tensor) {
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  if (status_ != base::kStatusCodeOk) return status_;
  if (!accepting_ || !tensor) return base::kStatusCodeErrorInvalidParam;
  for (auto *input : input_tensors_) {
    if (input->getName() != tensor->getName()) continue;
    // Pipeline shapes are fixed for the lifetime of a stage plan.
    if (input->getDataType() != tensor->getDataType() || input->getShape() != tensor->getShape()) {
      pending_inputs_.clear();
      return base::kStatusCodeErrorInvalidParam;
    }
    if (!queues_.count(input)) return base::kStatusCodeErrorInvalidParam;
    PipelineTensor::Packet frame;
    auto status = snapshot(tensor, frame);
    if (status == base::kStatusCodeOk) pending_inputs_[input] = std::move(frame);
    else pending_inputs_.clear();
    return status;
  }
  pending_inputs_.clear();
  return base::kStatusCodeErrorInvalidParam;
}

device::Tensor *PipelineRuntime::getOutputTensorAfterRun(const std::string &name,
    base::DeviceType device_type, bool is_copy, base::DataFormat) {
  device::Tensor *output = nullptr;
  for (auto *tensor : output_tensors_) if (tensor->getName() == name) { output = tensor; break; }
  auto it = queues_.find(output);
  if (it == queues_.end()) return nullptr;
  auto frame = it->second->popShared(nullptr);
  if (!frame) return nullptr;
  auto *device = device::getDevice(device_type);
  if (!device) return nullptr;
  if (is_copy || frame->getDevice() != device) {
    auto copy = std::unique_ptr<device::Tensor>(new device::Tensor(device, frame->getDesc(), name));
    if (frame->copyTo(copy.get()) != base::kStatusCodeOk) return nullptr;
    return copy.release();
  }
  // A borrowed result is valid until the next retrieval of this output or deinit.
  std::lock_guard<std::mutex> lock(pipeline_mutex_);
  borrowed_outputs_[name] = frame;
  return frame.get();
}

} // namespace net
} // namespace nndeploy
