
#include "nndeploy/source/inference/mnn/mnn_config.h"

namespace nndeploy {
namespace inference {

static TypeConfigRegister<TypeConfigCreator<MnnConfigImpl>>
    g_mnn_config_register(base::kInferenceTypeMnn);

base::Status MnnConfigImpl::jsonToConfig(const std::string &json,
                                         bool is_path) {
  base::Status status = DefaultConfigImpl::jsonToConfig(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status MnnConfigImpl::set(const std::string &key, base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  if (key == "library_path") {
    uint8_t *tmp = nullptr;
    if (value.get(&tmp)) {
      library_path_ = std::string(reinterpret_cast<char *>(tmp));
    } else {
      status = base::kStatusCodeErrorInvalidParam;
    }
  }
  return base::kStatusCodeOk;
}

MNNForwardType MnnConfigImpl::convertFromDeviceType(base::DeviceType src) {
  MNNForwardType type = MNN_FORWARD_CPU;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      type = MNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeX86:
      type = MNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeArm:
      type = MNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeOpenCL:
      type = MNN_FORWARD_OPENCL;
      break;
    case base::kDeviceTypeCodeOpenGL:
      type = MNN_FORWARD_OPENGL;
      break;
    case base::kDeviceTypeCodeMetal:
      type = MNN_FORWARD_METAL;
      break;
    case base::kDeviceTypeCodeCuda:
      type = MNN_FORWARD_CUDA;
      break;
    default:
      type = MNN_FORWARD_CPU;
      break;
  }
  return type;
}
MNN::BackendConfig::PowerMode MnnConfigImpl::convertFromPowerType(
    base::PowerType src) {
  switch (src) {
    case base::kPowerTypeLow:
      return MNN::BackendConfig::PowerMode::Power_Low;
    case base::kPowerTypeNormal:
      return MNN::BackendConfig::PowerMode::Power_Normal;
    case base::kPowerTypeHigh:
      return MNN::BackendConfig::PowerMode::Power_High;
    default:
      return MNN::BackendConfig::PowerMode::Power_Normal;
  }
}
MNN::BackendConfig::PrecisionMode MnnConfigImpl::convertFromPowerType(
    base::PrecisionType src) {
  switch (src) {
    case base::kPrecisionTypeFp16:
      return MNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeFpBFp16:
      return MNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeFp32:
      return MNN::BackendConfig::PrecisionMode::Precision_Normal;
    case base::kPrecisionTypeFp64:
      return MNN::BackendConfig::PrecisionMode::Precision_High;
    default:
      return MNN::BackendConfig::PrecisionMode::Precision_Normal;
  }
}

base::Status MnnConfigImpl::convertFromConfig(
    MnnConfigImpl *config, MNN::ScheduleConfig *internal_config) {
  if (config == nullptr || internal_config == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }

  internal_config->saveTensors = config->save_tensors_;
  internal_config->type = convertFromDeviceType(config->device_type_);
  if (internal_config->type == MNN_FORWARD_CPU) {
    internal_config->numThread = config->num_thread_;
  } else if (internal_config->type == MNN_FORWARD_OPENCL ||
             internal_config->type == MNN_FORWARD_OPENGL ||
             internal_config->type == MNN_FORWARD_METAL ||
             internal_config->type == MNN_FORWARD_CUDA) {
    internal_config->mode = config->gpu_tune_mode_;
  }
  internal_config->path = config->path_;
  internal_config->backupType =
      convertFromDeviceType(config->backup_device_type_);

  internal_config->backendConfig = new MNN::BackendConfig();
  internal_config->backendConfig->power =
      convertFromPowerType(config->power_type_);
  internal_config->backendConfig->precision =
      convertFromPowerType(config->precision_type_);
  internal_config->backendConfig->memory = config->memory_mode_;

  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
