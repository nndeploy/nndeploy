#pragma once

#include <cstdint>

// The permute tiling define, support at most 6 dimensions.
struct PermuteTilingData {
  // The input dims.
  uint32_t dim0;
  uint32_t dim1;
  uint32_t dim2;

  // The input strides.
  uint32_t stride0;
  uint32_t stride1;
  uint32_t stride2;

  // The new dim order.
  uint32_t new_idx0;
  uint32_t new_idx1;
  uint32_t new_idx2;

  // The strides for new tensor.
  uint32_t new_stride0;
  uint32_t new_stride1;
  uint32_t new_stride2;

  // The tiling block length and total length, num of elments.
  uint32_t total_length;

  // The used aicore number.
  uint32_t used_core_num;

  // The tiling key, used to specify data type.
  uint32_t tiling_key;
};
