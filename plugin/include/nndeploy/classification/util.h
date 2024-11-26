
#ifndef _NNDEPLOY_CLASSIFICATION_CLASSIFICATION_UTIL_H_
#define _NNDEPLOY_CLASSIFICATION_CLASSIFICATION_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/base/any.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace classification {

// topk sometimes is a very small value
// so this implementation is simple but I don't think it will
// cost too much time
// Also there may be cause problem since we suppose the minimum value is
// -99999999
// Do not use this function on array which topk contains value less than
// -99999999
template <typename T>
std::vector<int32_t> topKIndices(const T* array, int array_size, int topk) {
  topk = std::min(array_size, topk);
  std::vector<int32_t> res(topk);
  std::set<int32_t> searched;
  for (int32_t i = 0; i < topk; ++i) {
    T min = static_cast<T>(-99999999);
    for (int32_t j = 0; j < array_size; ++j) {
      if (searched.find(j) != searched.end()) {
        continue;
      }
      if (*(array + j) > min) {
        res[i] = j;
        min = *(array + j);
      }
    }
    searched.insert(res[i]);
  }
  return res;
}

}  // namespace classification
}  // namespace nndeploy

#endif /* _NNDEPLOY_CLASSIFICATION_CLASSIFICATION_COMMON_H_ */
