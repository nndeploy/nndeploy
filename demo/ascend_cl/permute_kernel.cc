#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "demo/ascend_cl/permute_tiling.h"

constexpr int32_t BUFFER_NUM = 1;

__aicore__ inline void CopyTiling(PermuteTilingData* tiling, GM_ADDR tiling_gm) {
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling);
  __gm__ uint32_t* tiling_gm_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling_gm);

  for (int i = 0; i < sizeof(PermuteTilingData) / sizeof(uint32_t); i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

template <typename T>
class PermuteKernel {
 public:
  __aicore__ inline PermuteKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, PermuteTilingData* tiling);

  __aicore__ inline void Process();

 private:
  // The process.
  __aicore__ inline uint32_t GetNewIndexPos(uint32_t i, uint32_t j);
  __aicore__ inline uint32_t GetNewIndexPos(uint32_t i, uint32_t j, uint32_t k);

  // The in/out stage process.
  __aicore__ inline void CopyIn(int32_t src_idx);
  __aicore__ inline void CopyOut(int32_t dst_idx);

  // The last dim is changed, loop every item.
  __aicore__ inline void ProcessSingle();

  // The last dim is not changed, loop through last dim.
  __aicore__ inline void ProcessBatch();

  // The tiling config.
  PermuteTilingData* tiling_;

  TPipe pipe_;
  GlobalTensor<T> input_gm_;
  GlobalTensor<T> output_gm_;

  // The input and output queue.
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue_;

  int32_t block_idx_ = 0;
  int32_t block_dim_ = 0;
};

template <typename T>
__aicore__ uint32_t PermuteKernel<T>::GetNewIndexPos(uint32_t i, uint32_t j) {
  uint32_t indexes[2] = {i, j};
  return (indexes[tiling_->new_idx0] * tiling_->new_stride0 + indexes[tiling_->new_idx1] * tiling_->new_stride1);
}

template <typename T>
__aicore__ uint32_t PermuteKernel<T>::GetNewIndexPos(uint32_t i, uint32_t j, uint32_t k) {
  uint32_t indexes[3] = {i, j, k};
  return (indexes[tiling_->new_idx0] * tiling_->new_stride0 + indexes[tiling_->new_idx1] * tiling_->new_stride1 +
          indexes[tiling_->new_idx2] * tiling_->new_stride2);
}

template <typename T>
__aicore__ void PermuteKernel<T>::Init(GM_ADDR input, GM_ADDR output, PermuteTilingData* tiling) {
  tiling_ = tiling;

  block_idx_ = GetBlockIdx();
  block_dim_ = tiling_->used_core_num;

  input_gm_.SetGlobalBuffer((__gm__ T*)input);
  output_gm_.SetGlobalBuffer((__gm__ T*)output);

  pipe_.InitBuffer(input_queue_, BUFFER_NUM, tiling_->dim2 * sizeof(T));
}

template <typename T>
__aicore__ inline void PermuteKernel<T>::CopyIn(int32_t src_idx) {
  LocalTensor<T> local_tensor = input_queue_.AllocTensor<T>();

  DataCopy(local_tensor, input_gm_[src_idx], tiling_->dim2);
  pipe_barrier(PIPE_ALL);

  input_queue_.EnQue(local_tensor);
}

template <typename T>
__aicore__ inline void PermuteKernel<T>::CopyOut(int32_t dst_idx) {
  LocalTensor<T> local_tensor = input_queue_.DeQue<T>();

  DataCopy(output_gm_[dst_idx], local_tensor, tiling_->dim2);
  pipe_barrier(PIPE_ALL);

  input_queue_.FreeTensor(local_tensor);
}

template <typename T>
__aicore__ inline void PermuteKernel<T>::ProcessSingle() {
  for (uint64_t idx = block_idx_; idx < tiling_->total_length; idx += tiling_->used_core_num) {
    uint64_t rest = 0;
    uint64_t i = idx / tiling_->stride0;
    rest = idx % tiling_->stride0;
    uint64_t j = rest / tiling_->stride1;
    rest = rest % tiling_->stride1;
    uint64_t k = rest % tiling_->stride1;

    uint64_t dst_pos = GetNewIndexPos(i, j, k);
    output_gm_.SetValue(dst_pos, input_gm_.GetValue(idx));
  }
}

template <typename T>
__aicore__ inline void PermuteKernel<T>::ProcessBatch() {
  for (uint64_t idx = block_idx_; idx < tiling_->total_length; idx += tiling_->used_core_num) {
    uint64_t rest = 0;
    uint64_t i = idx / tiling_->stride0;
    rest = idx % tiling_->stride0;
    uint64_t j = rest % tiling_->stride0;
    uint64_t dst_pos = GetNewIndexPos(i, j);

    CopyIn(idx * tiling_->dim2);
    CopyOut(dst_pos * tiling_->dim2);
  }
}

template <typename T>
__aicore__ void PermuteKernel<T>::Process() {
  if (tiling_->new_idx2 != 2) {
    ProcessSingle();
  } else {
    ProcessBatch();
  }
}

extern "C" __global__ __aicore__ void InvokePermuteKernel(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm) {
  PermuteTilingData tiling;
  CopyTiling(&tiling, tiling_gm);
  if (GetBlockIdx() >= tiling.used_core_num) {
    return;
  }

  if (tiling.tiling_key == 0) {
    PermuteKernel<half> op;
    op.Init(input, output, &tiling);
    op.Process();
  } else if (tiling.tiling_key == 1) {
    PermuteKernel<float> op;
    op.Init(input, output, &tiling);
    op.Process();
  }
}