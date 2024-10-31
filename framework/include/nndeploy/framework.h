
#ifndef _NNDEPLOY_FRAMEWORK_H_
#define _NNDEPLOY_FRAMEWORK_H_

#include "nndeploy/base/macro.h"

/**
 * @brief init framework
 *
 * @return int
 * @retval 0 success
 * @retval other failed
 */
NNDEPLOY_C_API int nndeployFrameworkInit();

/**
 * @brief deinit framework
 *
 * @return int
 * @retval 0 success
 * @retval other failed
 */
NNDEPLOY_C_API int nndeployFrameworkDeinit();

#endif
