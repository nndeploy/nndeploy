
#include "nndeploy/base/shape.h"

#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

size_t shapeCount(const IntVector &dims, int start_index, int end_index) {
  if (-1 == end_index || end_index > dims.size()) {
    end_index = static_cast<int>(dims.size());
  }

  size_t size = 1;
  for (int index = start_index; index < end_index; ++index) {
    size *= dims[index];
  }
  return size;
}

IntVector shapeMax(const IntVector &dims0, const IntVector &dims1,
                   int start_index, int end_index) {
  IntVector max_dims;
  IntVector small_dims;
  if (dims0.size() >= dims1.size()) {
    max_dims = dims0;
    small_dims = dims1;
  } else {
    max_dims = dims1;
    small_dims = dims0;
  }

  if (small_dims.size() <= start_index) {
    return max_dims;
  }

  if (-1 == end_index || end_index > small_dims.size()) {
    end_index = static_cast<int>(small_dims.size());
  }

  for (int i = start_index; i < end_index; i++) {
    max_dims[i] = std::max(max_dims[i], small_dims[i]);
  }

  return max_dims;
}

IntVector shapeMin(const IntVector &dims0, const IntVector &dims1,
                   int start_index, int end_index) {
  IntVector min_dims;
  IntVector small_dims;
  if (dims0.size() >= dims1.size()) {
    min_dims = dims0;
    small_dims = dims1;
  } else {
    min_dims = dims1;
    small_dims = dims0;
  }

  if (small_dims.size() <= start_index) {
    return small_dims;
  }

  if (-1 == end_index || end_index > small_dims.size()) {
    end_index = static_cast<int>(small_dims.size());
  }

  for (int i = start_index; i < end_index; i++) {
    min_dims[i] = std::min(min_dims[i], small_dims[i]);
  }

  return min_dims;
}

bool shapeEqual(const IntVector &dims0, const IntVector &dims1, int start_index,
                int end_index) {
  if (dims0.size() == 0 && dims1.size() == 0) {
    return true;
  }

  if (dims0.size() <= start_index) {
    return false;
  }

  if (-1 == end_index || end_index > dims0.size()) {
    end_index = static_cast<int>(dims0.size());
  }

  if (dims0.size() != dims1.size()) {
    return false;
  }

  for (int i = start_index; i < end_index; i++) {
    if (dims0[i] != dims1[i]) {
      return false;
    }
  }
  return true;
}

IntVector shapeNchw2Nhwc(const IntVector &dims) {
  NNDEPLOY_ASSERT(dims.size() == 4);
  const int n = dims[0];
  const int c = dims[1];
  const int h = dims[2];
  const int w = dims[3];
  std::vector<int> nhwc = {n, h, w, c};
  return nhwc;
}

IntVector shapeNhwc2Nchw(const IntVector &dims) {
  NNDEPLOY_ASSERT(dims.size() == 4);
  const int n = dims[0];
  const int h = dims[1];
  const int w = dims[2];
  const int c = dims[3];
  std::vector<int> nhwc = {n, c, h, w};
  return nhwc;
}

}  // namespace base
}  // namespace nndeploy
