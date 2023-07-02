
#ifndef _NNDEPLOY_BASE_OBJECT_H_
#define _NNDEPLOY_BASE_OBJECT_H_

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"

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

class NNDEPLOY_CC_API Movable {
 public:
  Movable() = default;
  Movable(const Movable&) = delete;
  Movable& operator=(const Movable&) = delete;
};

struct NNDEPLOY_CC_API Deleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      delete obj;
    }
  }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, Deleter>;

template <typename T>
class NNDEPLOY_CC_API Singleton {
 public:
  /**
  @brief get a reference to the singleton object
  */
  inline static T& getInstance() {
    static T instance;
    return instance;
  }

 private:
  Singleton() = default;
  ~Singleton() = default;
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;
  Singleton(Singleton&&) = delete;
  Singleton& operator=(Singleton&&) = delete;
};

}  // namespace base
}  // namespace nndeploy

#endif