#ifndef _NNDEPLOY_BASE_DLOPEN_H_
#define _NNDEPLOY_BASE_DLOPEN_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"

#ifdef WIN32
#define NOMINMAX
#include <libloaderapi.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nndeploy {
namespace base {

#define NNDEPLOY_DEFINE_FUNC_PTR(func) func##Func func = nullptr

#ifdef WIN32
#define NNDEPLOY_LOAD_FUNCTION_PTR(handle, func_name)                        \
  func_name =                                                                \
      reinterpret_cast<func_name##Func>(GetProcAddress(handle, #func_name)); \
  if (func_name == nullptr) {                                                \
    NNDEPLOY_LOGE("load func (%s) from (%s) failed!\n", #func_name,          \
                  library_path.c_str());                                     \
    return false;                                                            \
  }
// load function ptr use dlopen and dlsym. if cann't find func_name, that will
// be ok.
#define NNDEPLOY_TRY_LOAD_FUNCTION_PTR(handle, func_name)                    \
  func_name =                                                                \
      reinterpret_cast<func_name##Func>(GetProcAddress(handle, #func_name)); \
  if (func_name == nullptr) {                                                \
    NNDEPLOY_LOGE("load func (%s) from (%s) failed!\n", #func_name,          \
                  library_path.c_str());                                     \
  }
#else  // WIN32
// load function ptr use dlopen and dlsym. if cann't find func_name, will return
// false.
#define NNDEPLOY_LOAD_FUNCTION_PTR(handle, func_name)                         \
  if (is_pixel) {                                                             \
    func_name =                                                               \
        reinterpret_cast<func_name##Func>(loadOpenCLPointer(#func_name));     \
  } else {                                                                    \
    func_name = reinterpret_cast<func_name##Func>(dlsym(handle, #func_name)); \
  }                                                                           \
  if (func_name == nullptr) {                                                 \
    NNDEPLOY_LOGE("load func (%s) from (%s) failed!\n", #func_name,           \
                  library_path.c_str());                                      \
    return false;                                                             \
  }

// load function ptr use dlopen and dlsym. if cann't find func_name, that will
// be ok.
#define NNDEPLOY_TRY_LOAD_FUNCTION_PTR(handle, func_name)                     \
  if (is_pixel) {                                                             \
    func_name =                                                               \
        reinterpret_cast<func_name##Func>(loadOpenCLPointer(#func_name));     \
  } else {                                                                    \
    func_name = reinterpret_cast<func_name##Func>(dlsym(handle, #func_name)); \
  }                                                                           \
  if (func_name == nullptr) {                                                 \
    NNDEPLOY_LOGE("load func (%s) from (%s) failed!\n", #func_name,           \
                  library_path.c_str());                                      \
  }

#endif  // end of WIN32

struct Handle {
#ifdef WIN32
  HMODULE handle_;
#else
  void *handle_;
#endif
};

extern NNDEPLOY_CC_API bool loadLibraryFromPath(const std::string &path,
                                                bool update);
extern NNDEPLOY_CC_API bool freeLibrary(const std::string &path);
extern NNDEPLOY_CC_API Handle *getLibraryHandle(const std::string &path,
                                                bool update);

}  // namespace base
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASE_DLOPEN_H_ */