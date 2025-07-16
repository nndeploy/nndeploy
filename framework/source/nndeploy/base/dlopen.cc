#include "nndeploy/base/dlopen.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace base {

class NNDEPLOY_CC_API DlopenSingleton {
 public:
  static DlopenSingleton *GetInstance() {
    static std::once_flag dlopen_singleton_once;
    std::call_once(dlopen_singleton_once, []() {
      dlopen_singleton_.reset(new DlopenSingleton(),
                              [](DlopenSingleton *p) { delete p; });
    });
    return dlopen_singleton_.get();
  }

  bool loadLibraryFromPath(const std::string &path, bool update) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = handle_map_.find(path);
    if (iter != handle_map_.end()) {
      if (!update) {
        return true;
      }
      // 如果需要更新，先释放旧的库
      this->freeLibrary(path);
    }
    
    // 检查文件是否存在
    if (!exists(path)) {
      NNDEPLOY_LOGE("library file not exist: %s\n", path.c_str());
      return false;
    }
    
    Handle *handle = new Handle();
#ifdef WIN32
    handle->handle_ = LoadLibraryA(path.c_str());
    if (handle->handle_ == nullptr) {
      DWORD error = GetLastError();
      NNDEPLOY_LOGE("load library from path(%s) failed! Error code: %lu\n", path.c_str(), error);
      delete handle;
      return false;
    }
#else
    handle->handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle->handle_ == nullptr) {
      const char* error = dlerror();
      NNDEPLOY_LOGE("load library from path(%s) failed! Error: %s\n", path.c_str(), error ? error : "unknown error");
      delete handle;
      return false;
    }
#endif
    
    handle_map_[path] = handle;
    // NNDEPLOY_LOGI("successfully loaded library: %s\n", path.c_str());
    return true;
  }

  bool freeLibrary(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = handle_map_.find(path);
    if (iter == handle_map_.end()) {
      NNDEPLOY_LOGI("free library from path(%s) not found!\n", path.c_str());
      return true;
    }
    Handle *handle = iter->second;
#ifdef WIN32
    if (handle->handle_ != nullptr) {
      if (FreeLibrary(handle->handle_) == 0) {
        NNDEPLOY_LOGE("free library from path(%s) failed!\n", path.c_str());
        return false;
      }
    }
#else
    if (handle->handle_ != nullptr) {
      if (dlclose(handle->handle_) != 0) {
        NNDEPLOY_LOGE("free library from path(%s) failed!\n", path.c_str());
        return false;
      }
    }
#endif
    delete handle;
    handle_map_.erase(iter);
    return true;
  }

  Handle *getLibraryHandle(const std::string &path, bool update) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = handle_map_.find(path);
    
    // 如果库已存在且不需要更新，直接返回
    if (iter != handle_map_.end() && !update) {
      return iter->second;
    }
    
    // 加载或重新加载库
    bool ret = this->loadLibraryFromPath(path, update);
    if (!ret) {
      NNDEPLOY_LOGE("get library handle from path(%s) failed!\n", path.c_str());
      return nullptr;
    }
    
    // 返回加载后的句柄
    iter = handle_map_.find(path);
    return (iter != handle_map_.end()) ? iter->second : nullptr;
  }

  DlopenSingleton(const DlopenSingleton &) = delete;
  DlopenSingleton &operator=(const DlopenSingleton &) = delete;

 private:
  DlopenSingleton() = default;
  ~DlopenSingleton() {
    // std::lock_guard<std::mutex> lock(mutex_);
    for (auto &item : handle_map_) {
      this->freeLibrary(item.first);
    }
    handle_map_.clear();
  }

  static std::shared_ptr<DlopenSingleton> dlopen_singleton_;
  std::map<std::string, Handle *> handle_map_;
  std::mutex mutex_;
};

std::shared_ptr<DlopenSingleton> DlopenSingleton::dlopen_singleton_ = nullptr;

bool loadLibraryFromPath(const std::string &path, bool update) {
  if (!exists(path)) {
    NNDEPLOY_LOGE("path is empty");
    return false;
  }
  return DlopenSingleton::GetInstance()->loadLibraryFromPath(path, false);
}

bool freeLibrary(const std::string &path) {
  if (!exists(path)) {
    NNDEPLOY_LOGE("path is empty");
    return false;
  }
  return DlopenSingleton::GetInstance()->freeLibrary(path);
}

Handle *getLibraryHandle(const std::string &path, bool update) {
  if (!exists(path)) {
    NNDEPLOY_LOGE("path is empty");
    return nullptr;
  }
  return DlopenSingleton::GetInstance()->getLibraryHandle(path, update);
}

}  // namespace base
}  // namespace nndeploy