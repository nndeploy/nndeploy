/**
 * @file profiler.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_PROFILER_H_
#define _NNDEPLOY_INCLUDE_BASE_PROFILER_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"

namespace nndeploy {
namespace base {

int add(int a, int b) {
  return a+b;
}

}  // namespace base
}  // namespace nndeploy

#endif