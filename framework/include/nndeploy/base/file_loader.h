//
//  FileLoader.hpp
//  MNN
//
//  Created by MNN on 2019/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <mutex>
#include <vector>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

class NNDEPLOY_CC_API BaseLoader {
 public:
  BaseLoader() = default;
  virtual ~BaseLoader() = default;
  virtual bool read(char* buffer, int64_t size) = 0;
};

class NNDEPLOY_CC_API FileLoader : public BaseLoader {
 public:
  FileLoader(const char* file, bool init = false);

  ~FileLoader();

  bool read();

  static bool write(const char* filePath,
                    std::pair<const void*, size_t> cacheInfo);

  bool valid() const { return mFile != nullptr; }
  inline size_t size() const { return mTotalSize; }
  inline std::string path() const { return mFilePath; }

  // bool merge(AutoStorage<uint8_t>& buffer);

  int offset(int64_t offset);

  bool read(char* buffer, int64_t size);

 private:
  void _init();
  std::vector<std::pair<size_t, void*>> mBlocks;
  FILE* mFile = nullptr;
  static const int gCacheSize = 4096;
  size_t mTotalSize = 0;
  std::string mFilePath;
  bool mInited = false;
};

class NNDEPLOY_CC_API MemoryLoader : public BaseLoader {
 public:
  MemoryLoader(unsigned char* ptr) : buffer_(ptr) {}
  virtual bool read(char* dst, int64_t size) override {
    ::memcpy(dst, buffer_, size);
    buffer_ += size;
    return true;
  }

 private:
  unsigned char* buffer_ = nullptr;
};

}  // namespace base
}  // namespace nndeploy
