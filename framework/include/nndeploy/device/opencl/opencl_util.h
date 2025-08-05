#ifndef _NNDEPLOY_DEVICE_OPENCL_OPENCL_UTIL_H_
#define _NNDEPLOY_DEVICE_OPENCL_OPENCL_UTIL_H_

#include "nndeploy/device/opencl/opencl_include.h"

namespace nndeploy {
namespace device {
#define NNDEPLOY_OPENCL_FETAL_ERROR(err)                           \
  {                                                                \
    std::stringstream _where, _message;                            \
    _where << __FILE__ << ':' << __LINE__;                         \
    _message << std::string(err) + "\n"                            \
             << __FILE__ << ':' << __LINE__ << "\nAborting... \n"; \
    NNDEPLOY_LOGE("%s", _message.str().c_str());                   \
    exit(EXIT_FAILURE);                                            \
  }

#define NNDEPLOY_OPENCL_CHECK(status)                           \
  {                                                             \
    std::stringstream _error;                                   \
    if (CL_SUCCESS != status) {                                 \
      _error << "OpenCL failure " << status << ": " << (status) \
             << " " NNDEPLOY_OPENCL_FETAL_ERROR(_error.str());  \
    }                                                           \
  }

}  // namespace device
}  // namespace nndeploy

#endif