#pragma once

#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "nndeploy/demo/ascend_cl/permute_tiling.h"


// NOTE(karlluo): perform the same process as Pytorch, just change shape and stride
void Permute(const aclTensor* permute_input, void** permute_input_tensor_addr_ptr, aclTensor** permute_output,
             const std::vector<int64_t>& dims, aclrtStream& stream, void (*ws_func)(size_t, void**));

template <typename T>
class PermuteKernelWrapper {
 public:
  PermuteKernelWrapper();
  ~PermuteKernelWrapper();

  // Permute.
  void Forward(void* output, void* input, void* tiling, int block_dim, aclrtStream stream);
  void Forward(void* output, void* input, const std::vector<uint64_t>& shape, const std::vector<uint64_t> new_indexes,
               aclrtStream stream);

  // Cache tiling
  void CacheTiling(void* dev, size_t key, const std::vector<uint64_t>& shape, const std::vector<uint64_t> new_indexes,
                   aclrtStream stream);

  // Return device pointer of tiling struct.
  void* GetTilingData(size_t key, int& block_dim);

  size_t GetTilingSize() const { return tiling_size_; }

 private:
  // Copy the tiling data from host to global memory.
  void CopyTilingToDevice(aclrtStream stream);

  // Generate tiling for input shape and indexes.
  void GenerateTiling(const std::vector<uint64_t>& shape, const std::vector<uint64_t> new_indexes,
                      PermuteTilingData& tiling_data);

  // The tiling data for current request.
  PermuteTilingData tiling_data_;

  // The tiling buffer on global memory
  void* tiling_buffer_gm_;

  // The size of tiling data.
  size_t tiling_size_;

  // The tiling cache
  std::unordered_map<size_t, void*> tiling_cache_;

  // The tiling used cores.
  std::unordered_map<size_t, int> tiling_cores_;
};
