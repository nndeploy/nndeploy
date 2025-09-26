//
//  diskembedding.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DISKEMBEDDING_hpp
#define DISKEMBEDDING_hpp

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/file_loader.h"

namespace MNN {
namespace Transformer {

typedef void (*DequantFunction)(const uint8_t*, float*, float, float, int);

class DiskEmbedding {
 public:
  explicit DiskEmbedding(std::string fileName = "") {};
  ~DiskEmbedding() {}
  void embedding(const std::vector<int>& input_ids, float* ptr);

 private:
  void seek_read(uint8_t* dst, size_t size, size_t offset);
  std::unique_ptr<uint8_t[]> mAlpha = nullptr;
  std::unique_ptr<uint8_t[]> mWeight = nullptr;
  std::unique_ptr<base::FileLoader> mFile;
  DequantFunction mDequantFunc;
  int mHiddenSize, mTokenSize;
  float mOffset = 0.0f;
  bool mAsymc = true;
  int64_t mWeightOffset = 0;
  int64_t mBlockNum = 0, mQuantBlock = 0, mQuantBit = 0;
};

}  // namespace Transformer
}  // namespace MNN

#endif  // DISKEMBEDDING_hpp
