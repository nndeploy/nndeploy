#ifndef _NNDEPLOY_INFERENCE_TENSOR_RT_TENSOR_RT_UTIL_H_
#define _NNDEPLOY_INFERENCE_TENSOR_RT_TENSOR_RT_UTIL_H_

#include "nndeploy/inference/tensor_rt/tensor_rt_include.h"

namespace nndepoy {
namespace inference {

class TensorRtLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    // suppress info-level messages
#ifndef DEBUG
    if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;
#endif
    const char* skips[] = {
        "INVALID_ARGUMENT: Cannot find binding of given name",
        "Unused Input:",
        "Detected invalid timing cache",
        "unused or used only at compile-time",
    };

    std::string msg_str = std::string(msg);
    for (auto skip : skips) {
      if (msg_str.find(skip) != std::string::npos) {
        return;
      }
    }
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        break;
      case Severity::kVERBOSE:
        std::cerr << "VERBOSE: ";
        break;
      default:
        break;
    }
    std::cerr << msg << std::endl;
  }
};

}  // namespace inference
}  // namespace nndepoy

#endif /* _NNDEPLOY_INFERENCE_TENSOR_RT_TENSOR_RT_UTIL_H_ */
