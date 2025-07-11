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

  bool loadLibraryFromPath(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = handle_map_.find(path);
    if (iter != handle_map_.end()) {
      return true;
    }

    Handle *handle = new Handle();
#ifdef WIN32
    handle->handle_ = LoadLibraryA(path.c_str());
#else
    handle->handle_ = dlopen(path.c_str(), RTLD_NOW);
#endif
    if (handle->handle_ == nullptr) {
      NNDEPLOY_LOGE("load library from path(%s) failed!\n", path.c_str());
      delete handle;
      return false;
    }
    handle_map_[path] = handle;
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

  Handle *getLibraryHandle(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = handle_map_.find(path);
    if (iter != handle_map_.end()) {
      return iter->second;
    } else {
      NNDEPLOY_LOGE("get library handle from path(%s) not found!\n",
                    path.c_str());
      return nullptr;
    }
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

bool loadLibraryFromPath(const std::string &path) {
  if (!exists(path)) {
    NNDEPLOY_LOGE("path is empty");
    return false;
  }
  return DlopenSingleton::GetInstance()->loadLibraryFromPath(path);
}

bool freeLibrary(const std::string &path) {
  if (!exists(path)) {
    NNDEPLOY_LOGE("path is empty");
    return false;
  }
  return DlopenSingleton::GetInstance()->freeLibrary(path);
}

Handle *getLibraryHandle(const std::string &path) {
  if (!exists(path)) {
    NNDEPLOY_LOGE("path is empty");
    return nullptr;
  }
  return DlopenSingleton::GetInstance()->getLibraryHandle(path);
}

}  // namespace base
}  // namespace nndeploy