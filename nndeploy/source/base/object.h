
#ifndef _NNDEPLOY_SOURCE_BASE_OBJECT_H_
#define _NNDEPLOY_SOURCE_BASE_OBJECT_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/include_c_cpp.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace base {

class NNDEPLOY_CC_API NonCopyable {
 public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable(NonCopyable&&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable& operator=(NonCopyable&&) = delete;
};

}  // namespace base
}  // namespace nndeploy

#endif