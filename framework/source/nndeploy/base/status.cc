
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

Status::Status() : code_(kStatusCodeOk) {}
Status::Status(int code) : code_(code) {}
Status::Status(StatusCode code) : code_(code) {}
Status::~Status() {}

Status::Status(const Status &other) { code_ = other.code_; }
Status &Status::operator=(const Status &other) {
  code_ = other.code_;
  return *this;
};
Status &Status::operator=(const StatusCode &other) {
  code_ = other;
  return *this;
};
Status &Status::operator=(int other) {
  code_ = other;
  return *this;
};

Status::Status(Status &&other) = default;
Status &Status::operator=(Status &&other) = default;

bool Status::operator==(const Status &other) const {
  if (code_ == other.code_) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator==(const StatusCode &other) const {
  if (code_ == other) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator==(int other) const {
  if (code_ == other) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(const Status &other) const {
  if (code_ != other.code_) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(const StatusCode &other) const {
  if (code_ != other) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int other) const {
  if (code_ != other) {
    return true;
  } else {
    return false;
  }
};

Status::operator int() const { return code_; }

Status::operator bool() const { return code_ == kStatusCodeOk; }

Status Status::operator+(const Status &other) {
  if (code_ == kStatusCodeOk) {
    code_ = other.code_;
  }
  return *this;
}

std::string Status::desc() const {
  std::string str = statusCodeToString(static_cast<StatusCode>(code_));
  // switch (code_) {
  //   default:
  //     str = std::to_string(static_cast<int>(code_));
  //     break;
  // }
  return str;
};

StatusCode Status::getStatusCode() const {
  return static_cast<StatusCode>(code_);
}

std::string statusCodeToString(StatusCode code) {
  switch (code) {
    case kStatusCodeOk:
      return "kStatusCodeOk";
    case kStatusCodeErrorUnknown:
      return "kStatusCodeErrorUnknown";
    case kStatusCodeErrorOutOfMemory:
      return "kStatusCodeErrorOutOfMemory";
    case kStatusCodeErrorNotSupport:
      return "kStatusCodeErrorNotSupport";
    case kStatusCodeErrorNotImplement:
      return "kStatusCodeErrorNotImplement";
    case kStatusCodeErrorInvalidValue:
      return "kStatusCodeErrorInvalidValue";
    case kStatusCodeErrorInvalidParam:
      return "kStatusCodeErrorInvalidParam";
    case kStatusCodeErrorNullParam:
      return "kStatusCodeErrorNullParam";
    case kStatusCodeErrorThreadPool:
      return "kStatusCodeErrorThreadPool";
    case kStatusCodeErrorIO:
      return "kStatusCodeErrorIO";
    case kStatusCodeErrorDeviceCpu:
      return "kStatusCodeErrorDeviceCpu";
    case kStatusCodeErrorDeviceArm:
      return "kStatusCodeErrorDeviceArm";
    case kStatusCodeErrorDeviceX86:
      return "kStatusCodeErrorDeviceX86";
    case kStatusCodeErrorDeviceRiscV:
      return "kStatusCodeErrorDeviceRiscV";
    case kStatusCodeErrorDeviceCuda:
      return "kStatusCodeErrorDeviceCuda";
    case kStatusCodeErrorDeviceRocm:
      return "kStatusCodeErrorDeviceRocm";
    case kStatusCodeErrorDeviceSyCL:
      return "kStatusCodeErrorDeviceSyCL";
    case kStatusCodeErrorDeviceOpenCL:
      return "kStatusCodeErrorDeviceOpenCL";
    case kStatusCodeErrorDeviceOpenGL:
      return "kStatusCodeErrorDeviceOpenGL";
    case kStatusCodeErrorDeviceMetal:
      return "kStatusCodeErrorDeviceMetal";
    case kStatusCodeErrorDeviceVulkan:
      return "kStatusCodeErrorDeviceVulkan";
    case kStatusCodeErrorDeviceHexagon:
      return "kStatusCodeErrorDeviceHexagon";
    case kStatusCodeErrorDeviceMtkVpu:
      return "kStatusCodeErrorDeviceMtkVpu";
    case kStatusCodeErrorDeviceAscendCL:
      return "kStatusCodeErrorDeviceAscendCL";
    case kStatusCodeErrorDeviceAppleNpu:
      return "kStatusCodeErrorDeviceAppleNpu";
    case kStatusCodeErrorDeviceRkNpu:
      return "kStatusCodeErrorDeviceRkNpu";
    case kStatusCodeErrorDeviceQualcommNpu:
      return "kStatusCodeErrorDeviceQualcommNpu";
    case kStatusCodeErrorDeviceMtkNpu:
      return "kStatusCodeErrorDeviceMtkNpu";
    case kStatusCodeErrorDeviceSophonNpu:
      return "kStatusCodeErrorDeviceSophonNpu";
    case kStatusCodeErrorOpAscendCL:
      return "kStatusCodeErrorOpAscendCL";
    case kStatusCodeErrorInferenceDefault:
      return "kStatusCodeErrorInferenceDefault";
    case kStatusCodeErrorInferenceOpenVino:
      return "kStatusCodeErrorInferenceOpenVino";
    case kStatusCodeErrorInferenceTensorRt:
      return "kStatusCodeErrorInferenceTensorRt";
    case kStatusCodeErrorInferenceCoreML:
      return "kStatusCodeErrorInferenceCoreML";
    case kStatusCodeErrorInferenceTfLite:
      return "kStatusCodeErrorInferenceTfLite";
    case kStatusCodeErrorInferenceOnnxRuntime:
      return "kStatusCodeErrorInferenceOnnxRuntime";
    case kStatusCodeErrorInferenceAscendCL:
      return "kStatusCodeErrorInferenceAscendCL";
    case kStatusCodeErrorInferenceNcnn:
      return "kStatusCodeErrorInferenceNcnn";
    case kStatusCodeErrorInferenceTnn:
      return "kStatusCodeErrorInferenceTnn";
    case kStatusCodeErrorInferenceMnn:
      return "kStatusCodeErrorInferenceMnn";
    case kStatusCodeErrorInferencePaddleLite:
      return "kStatusCodeErrorInferencePaddleLite";
    case kStatusCodeErrorInferenceRknn:
      return "kStatusCodeErrorInferenceRknn";
    case kStatusCodeErrorInferenceTvm:
      return "kStatusCodeErrorInferenceTvm";
    case kStatusCodeErrorInferenceAITemplate:
      return "kStatusCodeErrorInferenceAITemplate";
    case kStatusCodeErrorInferenceSnpe:
      return "kStatusCodeErrorInferenceSnpe";
    case kStatusCodeErrorInferenceQnn:
      return "kStatusCodeErrorInferenceQnn";
    case kStatusCodeErrorInferenceSophon:
      return "kStatusCodeErrorInferenceSophon";
    case kStatusCodeErrorInferenceTorch:
      return "kStatusCodeErrorInferenceTorch";
    case kStatusCodeErrorInferenceTensorFlow:
      return "kStatusCodeErrorInferenceTensorFlow";
    case kStatusCodeErrorInferenceNeuroPilot:
      return "kStatusCodeErrorInferenceNeuroPilot";
    case kStatusCodeErrorDag:
      return "kStatusCodeErrorDag";
    default:
      return "Unknown Status Code";
  }
}

}  // namespace base
}  // namespace nndeploy
