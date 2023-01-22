/**
 * @file object.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 * @note
 * # ref tvm and mnn
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_OBJECT_H_
#define _NNDEPLOY_INCLUDE_BASE_OBJECT_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/basic.h"

namespace nndeploy {
namespace base {

class NonCopyable {
 public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable(const NonCopyable&&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&&) = delete;
};

}  // namespace base
}  // namespace nndeploy

#endif