
#ifndef _NNDEPLOY_SOURCE_DEVICE_CUDA_CUDA_UTIL_H_
#define _NNDEPLOY_SOURCE_DEVICE_CUDA_CUDA_UTIL_H_

#define CUDA_FETAL_ERROR(err)                                      \
  {                                                                \
    std::stringstream _where, _message;                            \
    _where << __FILE__ << ':' << __LINE__;                         \
    _message << std::string(err) + "\n"                            \
             << __FILE__ << ':' << __LINE__ << "\nAborting... \n"; \
    NNDEPLOY_LOGE("%s", _message.str().c_str());                   \
    exit(EXIT_FAILURE);                                            \
  }

#define CUDA_CHECK(status)                                          \
  {                                                                 \
    std::stringstream _error;                                       \
    if (cudaSuccess != status) {                                    \
      _error << "Cuda failure: " << cudaGetErrorName(status) << " " \
             << cudaGetErrorString(status);                         \
      CUDA_FETAL_ERROR(_error.str());                               \
    }                                                               \
  }

#define CUDNN_CHECK(status)                                       \
  {                                                               \
    std::stringstream _error;                                     \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      _error << "CUDNN failure: " << cudnnGetErrorString(status); \
      CUDA_FETAL_ERROR(_error.str());                             \
    }                                                             \
  }

#define CUBLAS_CHECK(status)               \
  {                                        \
    std::stringstream _error;              \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      _error << "Cublas failure: "         \
             << " " << status;             \
      CUDA_FETAL_ERROR(_error.str());      \
    }                                      \
  }

#endif /* _NNDEPLOY_SOURCE_DEVICE_CUDA_CUDA_UTIL_H_ */
