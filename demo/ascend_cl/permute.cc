/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "nndeploy/demo/ascend_cl/permute.h"
#include <type_traits>
#include "aclrtlaunch_InvokePermuteKernel.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/demo/ascend_cl/tiling_data_types.h"


// The max used ai core number.
constexpr uint32_t MAX_USED_CORE_NUM = 24;

// The min block size of permute.
constexpr uint32_t MIN_PERMUTE_BLOCK_SIZE = 16 * 16;

void Permute(const aclTensor* permute_input, void** permute_input_tensor_addr_ptr, aclTensor** permute_output,
             const std::vector<int64_t>& dims, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  int64_t* input_t_shape_ptr = nullptr;
  uint64_t input_t_dims_num = 0;
  ACL_CHECK_RET(aclGetViewShape(permute_input, &input_t_shape_ptr, &input_t_dims_num));
  std::vector<int64_t> input_t_shape(input_t_dims_num);
  for (uint64_t i = 0; i < input_t_dims_num; ++i) {
    input_t_shape[i] = input_t_shape_ptr[i];
  }
  std::vector<int64_t> input_t_strides;
  CalShapeStrides(input_t_shape, input_t_strides);

  std::vector<int64_t> output_t_shape(input_t_dims_num, 0);
  std::vector<int64_t> output_t_strides(input_t_shape.size(), 1);
  std::copy(input_t_shape.begin(), input_t_shape.end(), output_t_shape.begin());
  for (uint64_t i = 0; i < dims.size(); ++i) {
    output_t_shape[i] = input_t_shape[dims[i]];
    output_t_strides[i] = input_t_strides[dims[i]];
  }
  aclDataType acl_dtype;
  ACL_CHECK_RET(aclGetDataType(permute_input, &acl_dtype));
  *permute_output = aclCreateTensor(output_t_shape.data(), output_t_shape.size(), acl_dtype, output_t_strides.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, output_t_shape.data(), output_t_shape.size(),
                                    *permute_input_tensor_addr_ptr);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template <typename T>
PermuteKernelWrapper<T>::PermuteKernelWrapper() {
  tiling_size_ = sizeof(PermuteTilingData);
  ACL_CHECK_RET(aclrtMalloc(&tiling_buffer_gm_, tiling_size_, ACL_MEM_MALLOC_HUGE_FIRST));
}

template <typename T>
PermuteKernelWrapper<T>::~PermuteKernelWrapper() {
  ACL_CHECK_RET(aclrtFree(tiling_buffer_gm_));
}

template <typename T>
void PermuteKernelWrapper<T>::CopyTilingToDevice(aclrtStream stream) {
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_buffer_gm_, tiling_size_, &tiling_data_, tiling_size_,
                                 aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template <typename T>
void PermuteKernelWrapper<T>::GenerateTiling(const std::vector<uint64_t>& shape,
                                             const std::vector<uint64_t> new_indexes, PermuteTilingData& tiling_data) {
  std::vector<uint32_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  std::vector<int64_t> new_shape;
  for (auto i : new_indexes) {
    new_shape.push_back(shape[i]);
  }

  std::vector<uint32_t> new_strides(new_shape.size(), 1);
  for (int64_t i = new_shape.size() - 2; i >= 0; i--) {
    new_strides[i] = new_shape[i + 1] * new_strides[i + 1];
  }

  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }

  tiling_data.dim0 = shape[0];
  tiling_data.dim1 = shape[1];
  tiling_data.dim2 = shape[2];

  tiling_data.stride0 = strides[0];
  tiling_data.stride1 = strides[1];
  tiling_data.stride2 = strides[2];

  tiling_data.new_idx0 = new_indexes[0];
  tiling_data.new_idx1 = new_indexes[1];
  tiling_data.new_idx2 = new_indexes[2];

  tiling_data.new_stride0 = new_strides[0];
  tiling_data.new_stride1 = new_strides[1];
  tiling_data.new_stride2 = new_strides[2];

  // Use single core to work around with ascend's cache line align.
  if (tiling_data.new_idx2 != 2) {
    tiling_data.total_length = shape_size;
    tiling_data.used_core_num = 1;
  } else {
    tiling_data.total_length = shape_size / tiling_data.dim2;
    tiling_data.used_core_num = MAX_USED_CORE_NUM * 2;

    tiling_data.stride0 /= tiling_data.dim2;
    tiling_data.stride1 /= tiling_data.dim2;
    tiling_data.stride2 /= tiling_data.dim2;

    tiling_data.new_stride0 /= tiling_data.dim2;
    tiling_data.new_stride1 /= tiling_data.dim2;
    tiling_data.new_stride2 /= tiling_data.dim2;
  }

  if (sizeof(T) == 2) {
    tiling_data.tiling_key = static_cast<uint32_t>(TilingDataType::FLOAT16);
  } else if (sizeof(T) == 4) {
    tiling_data.tiling_key = static_cast<uint32_t>(TilingDataType::FLOAT32);
  }
}

template <typename T>
void PermuteKernelWrapper<T>::CacheTiling(void* dev, size_t key, const std::vector<uint64_t>& shape,
                                          const std::vector<uint64_t> new_indexes, aclrtStream stream) {
  PermuteTilingData tiling_data;
  GenerateTiling(shape, new_indexes, tiling_data);

  ACL_CHECK_RET(aclrtMemcpyAsync(dev, tiling_size_, &tiling_data, tiling_size_,
                                 aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  tiling_cache_[key] = dev;
  tiling_cores_[key] = tiling_data.used_core_num;
}

template <typename T>
void* PermuteKernelWrapper<T>::GetTilingData(size_t key, int& block_dim) {
  if (tiling_cache_.find(key) != tiling_cache_.end()) {
    block_dim = tiling_cores_.at(key);
    return tiling_cache_.at(key);
  }
  return nullptr;
}

template <typename T>
void PermuteKernelWrapper<T>::Forward(void* output, void* input, void* tiling, int block_dim, aclrtStream stream) {
  ACLRT_LAUNCH_KERNEL(InvokePermuteKernel)(block_dim, stream, input, output, tiling);
}

template <typename T>
void PermuteKernelWrapper<T>::Forward(void* output, void* input, const std::vector<uint64_t>& shape,
                                      const std::vector<uint64_t> new_indexes, aclrtStream stream) {
  GenerateTiling(shape, new_indexes, tiling_data_);
  CopyTilingToDevice(stream);

  ACLRT_LAUNCH_KERNEL(InvokePermuteKernel)(tiling_data_.used_core_num, stream, input, output, tiling_buffer_gm_);
}

template class PermuteKernelWrapper<aclFloat16>;
template class PermuteKernelWrapper<float>;
