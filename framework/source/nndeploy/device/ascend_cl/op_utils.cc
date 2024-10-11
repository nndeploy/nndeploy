#include "nndeploy/op/ascend_cl/op_utils.h"

#include <vector>

void CalShapeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides) {
  strides.resize(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
}