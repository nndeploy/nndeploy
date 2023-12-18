
#ifndef _NNDEPLOY_DEVICE_MDC_ASCEND_CL_UTIL_H_
#define _NNDEPLOY_DEVICE_MDC_ASCEND_CL_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/ascend_cl/ascend_cl_include.h"

namespace nndeploy {
namespace device {

#define NNDEPLOY_MDC_CHECK(status)                                      \
  do {                                                                  \
    aclError __ret = status;                                            \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);
/**
 * @brief queries the number of available devices
 */
inline uint32_t mdcGetNumDevices() {
  uint32_t n = 0;
  aclError ret = aclrtGetDeviceCount(&n);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtGetDeviceCount failed, errorCode is %d", ret);
  }
  return n;
}
/**
 * @brief gets the current device associated with the caller thread
 */
inline int mdcGetDeviceId() {
  int id;
  aclError ret = aclrtGetDevice(&id);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtGetDevice failed, errorCode is %d", ret);
  }
  return id;
}

}  // namespace device
}  // namespace nndeploy

#endif /* _NNDEPLOY_DEVICE_MDC_ASCEND_CL_UTIL_H_ */
