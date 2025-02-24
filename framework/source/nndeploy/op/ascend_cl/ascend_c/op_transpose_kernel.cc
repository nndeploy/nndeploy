#include "op_transpose_kernel.h"

#include "kernel_operator.h"

__aicore__ inline void CopyTiling(nndeploy::op::TransposeTilingData *tiling,
                                  GM_ADDR tiling_gm) {
  uint32_t *tiling_ptr = reinterpret_cast<uint32_t *>(tiling);
  __gm__ uint32_t *tiling_gm_ptr =
      reinterpret_cast<__gm__ uint32_t *>(tiling_gm);

  for (int i = 0;
       i < sizeof(nndeploy::op::TransposeTilingData) / sizeof(uint32_t);
       i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}
template <typename T>
class KernelTranspose {
 public:
  __aicore__ inline KernelTranspose() {}
  __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR tiling_gm) {
    CopyTiling(&tiling_, tiling_gm);
    N = tiling_.N;
    C = tiling_.C;
    H = tiling_.H;
    W = tiling_.W;
    inputSize = N * C * H * W;
    tmpBufferSize = (C + 2) * 16 * 16;
    srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
    dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
    pipe.InitBuffer(inQueueSrcVecIn, 1, inputSize * sizeof(T));
    pipe.InitBuffer(inQueueSrcVecOut, 1, inputSize * sizeof(T));
    pipe.InitBuffer(tmpQueue, 1, tmpBufferSize * sizeof(T));
  }
  __aicore__ inline void Process() {
    CopyIn();
    Compute();
    CopyOut();
  }

 private:
  __aicore__ inline void CopyIn() {
    AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
    AscendC::DataCopy(srcLocal, srcGlobal, inputSize);
    inQueueSrcVecIn.EnQue(srcLocal);
  }
  __aicore__ inline void Compute() {
    AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
    AscendC::LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();
    AscendC::LocalTensor<uint8_t> stackBuffer = tmpQueue.AllocTensor<uint8_t>();

    AscendC::TransposeParamsExt transposeParams;
    transposeParams.nSize = N;
    transposeParams.cSize = C;
    transposeParams.hSize = H;
    transposeParams.wSize = W;
    transposeParams.transposeType = transposetype;
    AscendC::Transpose(dstLocal, srcLocal, stackBuffer, transposeParams);
    inQueueSrcVecOut.EnQue<T>(dstLocal);
    inQueueSrcVecIn.FreeTensor(srcLocal);
    tmpQueue.FreeTensor(stackBuffer);
  }
  __aicore__ inline void CopyOut() {
    AscendC::LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
    AscendC::DataCopy(dstGlobal, dstLocal, inputSize);
    inQueueSrcVecOut.FreeTensor(dstLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueSrcVecIn;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> inQueueSrcVecOut;
  AscendC::TQue<AscendC::QuePosition::VECCALC, 1> tmpQueue;

  AscendC::GlobalTensor<T> srcGlobal;
  AscendC::GlobalTensor<T> dstGlobal;
  uint32_t N;
  uint32_t C;
  uint32_t H;
  uint32_t W;
  uint32_t inputSize, tmpBufferSize;
  AscendC::TransposeType transposetype =
      AscendC::TransposeType::TRANSPOSE_NCHW2NHWC;
  nndeploy::op::TransposeTilingData tiling_;
};

extern "C" __global__ __aicore__ void transpose(GM_ADDR srcGm, GM_ADDR dstGm,
                                                GM_ADDR tiling_gm) {
  KernelTranspose<half> op;
  op.Init(srcGm, dstGm, tiling_gm);
  op.Process();
}
