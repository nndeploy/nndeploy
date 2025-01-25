#include "op_mul_kernel.h"

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_NUM = 16;

__aicore__ inline void CopyTiling(nndeploy::op::MulCustomTilingData *tiling,
                                  GM_ADDR tiling_gm) {
  uint32_t *tiling_ptr = reinterpret_cast<uint32_t *>(tiling);
  __gm__ uint32_t *tiling_gm_ptr =
      reinterpret_cast<__gm__ uint32_t *>(tiling_gm);

  for (int i = 0;
       i < sizeof(nndeploy::op::MulCustomTilingData) / sizeof(uint32_t);
       i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

class KernelMul {
 public:
  __aicore__ inline KernelMul() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR tiling_gm) {
    CopyTiling(&tiling_, tiling_gm);
    uint32_t totalLength = tiling_.totalLength;

    uint32_t blockDim = AscendC::GetBlockNum();
    uint32_t totalLengthAligned =
        ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t bodyLength =
        (totalLengthAligned / blockDim + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    uint32_t tailLength =
        (totalLengthAligned / blockDim) / ALIGN_NUM * ALIGN_NUM;
    uint32_t bodyNum = (totalLengthAligned / ALIGN_NUM) % blockDim;
    uint32_t tailNum = blockDim - bodyNum;

    if (AscendC::GetBlockIdx() < bodyNum) {
      this->tileLength = bodyLength;
      xGm.SetGlobalBuffer(
          (__gm__ half *)x + bodyLength * AscendC::GetBlockIdx(), bodyLength);
      yGm.SetGlobalBuffer(
          (__gm__ half *)y + bodyLength * AscendC::GetBlockIdx(), bodyLength);
      zGm.SetGlobalBuffer(
          (__gm__ half *)z + bodyLength * AscendC::GetBlockIdx(), bodyLength);
    } else {
      this->tileLength = tailLength;
      xGm.SetGlobalBuffer((__gm__ half *)x + bodyLength * bodyNum +
                              tailLength * (AscendC::GetBlockIdx() - bodyNum),
                          tailLength);
      yGm.SetGlobalBuffer((__gm__ half *)y + bodyLength * bodyNum +
                              tailLength * (AscendC::GetBlockIdx() - bodyNum),
                          tailLength);
      zGm.SetGlobalBuffer((__gm__ half *)z + bodyLength * bodyNum +
                              tailLength * (AscendC::GetBlockIdx() - bodyNum),
                          tailLength);
    }

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
  }

  __aicore__ inline void Process() {
    CopyIn();
    Compute();
    CopyOut();
  }

 private:
  __aicore__ inline void CopyIn() {
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    AscendC::DataCopy(xLocal, xGm, this->tileLength);
    AscendC::DataCopy(yLocal, yGm, this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute() {
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue<half>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut() {
    AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    AscendC::DataCopy(zGm, zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<half> xGm;
  AscendC::GlobalTensor<half> yGm;
  AscendC::GlobalTensor<half> zGm;

  uint32_t tileLength;
  nndeploy::op::MulCustomTilingData tiling_;
};

extern "C" __global__ __aicore__ void mul_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z, GM_ADDR tiling_gm) {
  KernelMul op;
  op.Init(x, y, z, tiling_gm);
  op.Process();
}

namespace nndeploy {
namespace op {

void mul_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
                   uint8_t *z, MulCustomTilingData *data) {
  // mul_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}

}  // namespace op
}  // namespace nndeploy