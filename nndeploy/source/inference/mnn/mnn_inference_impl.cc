
#include "nndeploy/source/inference/mnn/mnn_inference_impl.h"

#include "nndeploy/source/base/shape.h"
namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<MnnInferenceImpl>>
    g_mnn_inference_register(base::kInferenceTypeMnn);

MnnInferenceImpl::~MnnInferenceImpl() {}

base::Status MnnInferenceImpl::init(const Config &config) {
  /**
   * @brief
   * @note
   * # Config -> MNN::ScheduleConfig
   * # 模型解析
   * # 能不能写入静态形状？
   */
  base::Status status = base::kStatusCodeOk;

  config_ = config;
  if (config.config_impl_->is_path_) {
    mnn_interpreter_ = MNN::Interpreter::createFromFile(
        config.config_impl_->model_value_[0].c_str());
  } else {
    mnn_interpreter_ = MNN::Interpreter::createFromBuffer(
        config.config_impl_->model_value_[0].c_str(),
        config.config_impl_->model_value_[0].length());
  }

  if (mnn_interpreter_ == nullptr) {
    return base::kStatusCodeErrorInferenceMnn;
  }

  return status;
}

base::Status MnnInferenceImpl::deinit() {
  if (mnn_interpreter_ != nullptr) {
    MNN::Interpreter::destroy(mnn_interpreter_);
  }
  return base::kStatusCodeOk;
}

base::Status MnnInferenceImpl::preRun(
    base::ShapeMap min_shape = base::ShapeMap(),
    base::ShapeMap opt_shape = base::ShapeMap(),
    base::ShapeMap max_shape = base::ShapeMap()) {
  base::Status status = base::kStatusCodeOk;

  min_shape_ = min_shape;
  opt_shape_ = opt_shape;
  max_shape_ = max_shape;

  status = convertConfigInternal();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  mnn_session_ = mnn_interpreter_->createSession(mnn_config_);
  current_shape_ = getInputShapeMapInternal();
  if (!max_shape.empty()) {
    reShape(max_shape);
  }

  status = createInputsOutputsInternal();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  return status;
}

base::Status MnnInferenceImpl::postRun() {
  bool third_status = mnn_interpreter_->releaseSession(mnn_session_);
  if (third_status) {
    return base::kStatusCodeOk;
  } else {
    return base::kStatusCodeErrorInferenceMnn;
  }
}

base::Status MnnInferenceImpl::reShape(base::ShapeMap &shape_map) {
  bool flag = false;
  for (auto iter : shape_map) {
    auto tmp = current_shape_.find(iter.first);
    if (tmp != current_shape_.end()) {
      auto &shape = current_shape_[iter.first];
      if (base::shapeEqual(iter.second, shape)) {
        continue;
      } else {
        // TODO：map的使用
        shape = iter.second;
        MNN::Tensor *tensor =
            mnn_interpreter_->getSessionInput(mnn_session_, iter.first.c_str());
        mnn_interpreter_->resizeTensor(tensor, shape);
        flag = true;
        // TODO：current shape也需要修改
      }
    } else {
      // TODO：add log
      return base::kStatusCodeErrorInferenceMnn;
    }
  }

  if (flag) {
    mnn_interpreter_->resizeSession(mnn_session_);
  }

  return base::kStatusCodeOk;
}

int64_t MnnInferenceImpl::getMemorySize() {
  MNN::Interpreter::SessionInfoCode code =
      MNN::Interpreter::SessionInfoCode::MEMORY;
  float fsize = 0;
  bool third_status =
      mnn_interpreter_->getSessionInfo(mnn_session_, code, &fsize);
  int64_t size = (int64_t)(fsize * 1024 * 1024);
  return size;
}

base::Status MnnInferenceImpl::setInputTensor(
    const std::string &name,
    const std::shared_ptr<device::Tensor> input_tensor) {}
//
std::shared_ptr<device::Tensor> MnnInferenceImpl::getOutputTensor(
    const std::string &name, std::vector<int32_t> config) {}

base::Status MnnInferenceImpl::run() {
  MNN::ErrorCode third_status = mnn_interpreter_->runSession(mnn_session_);
  if (third_status != MNN::NO_ERROR) {
    return base::kStatusCodeErrorInferenceMnn;
  } else {
    return base::kStatusCodeOk;
  }
}
base::Status MnnInferenceImpl::asyncRun() { return this->run(); }

MNN::ScheduleConfig MnnInferenceImpl::getMnnConfig() { return mnn_config_; }
MNN::Interpreter *MnnInferenceImpl::getMnnInterpreter() {
  return mnn_interpreter_;
}
MNN::Session *MnnInferenceImpl::getMnnSession() { return mnn_session_; }

base::ShapeMap MnnInferenceImpl::getInputShapeMapInternal() {
  const std::map<std::string, MNN::Tensor *> &mnn_input_tensors =
      mnn_interpreter_->getSessionInputAll(mnn_session_);
  base::ShapeMap shape;
  for (auto iter : mnn_input_tensors) {
    std::string name = iter.first;
    base::IntVector dims = iter.second->shape();
    if (dims[0] != 1) {
      dims[0] = 1;
      mnn_interpreter_->resizeTensor(iter.second, dims);
      mnn_interpreter_->resizeSession(mnn_session_);
    }
    shape.insert({name, dims});
  }
  return shape;
}

base::Status MnnInferenceImpl::createInputsOutputsInternal() {
  const std::map<std::string, MNN::Tensor *> &mnn_input_tensors =
      mnn_interpreter_->getSessionInputAll(mnn_session_);
  for (auto iter : mnn_input_tensors) {
    std::string name = iter.first;
    MNN::Tensor *mnn_input_tensor = iter.second;
    std::shared_ptr<device::Tensor> max_input_tensor;
    max_input_tensor.reset(new device::Tensor());
    // TODO
    max_input_tensor.create();
    max_input_tensors_.insert({name, max_input_tensor});
    std::shared_ptr<device::Tensor> current_input_tensor;
    current_input_tensor.reset(new device::Tensor());
    // TODO
    current_input_tensor.create();
    current_input_tensors_.insert({name, max_input_tensor});
  }
  const std::map<std::string, MNN::Tensor *> &mnn_output_tensors =
      mnn_interpreter_->getSessionOutputAll(mnn_session_);
  for (auto iter : mnn_output_tensors) {
    std::string name = iter.first;
    MNN::Tensor *mnn_input_tensor = iter.second;
    std::shared_ptr<device::Tensor> max_output_tensor;
    max_output_tensor.reset(new device::Tensor());
    // TODO
    max_output_tensor.create();
    max_output_tensors_.insert({name, max_output_tensor});
    std::shared_ptr<device::Tensor> current_output_tensor;
    current_output_tensor.reset(new device::Tensor());
    // TODO
    current_output_tensor.create();
    current_output_tensors_.insert({name, max_output_tensor});
  }

  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
