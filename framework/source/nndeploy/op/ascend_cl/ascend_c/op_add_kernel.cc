#include "op_add_kernel.h"

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline void CopyTiling(nndeploy::op::AddCustomTilingData *tiling,
                                  GM_ADDR tiling_gm) {
  uint32_t *tiling_ptr = reinterpret_cast<uint32_t *>(tiling);
  __gm__ uint32_t *tiling_gm_ptr =
      reinterpret_cast<__gm__ uint32_t *>(tiling_gm);

  for (int i = 0;
       i < sizeof(nndeploy::op::AddCustomTilingData) / sizeof(uint32_t);
       i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR tiling_gm) {
    CopyTiling(&tiling_, tiling_gm);
    this->blockLength = tiling_.totalLength;
    this->tileNum = tiling_.tileNum;
    this->tileLength = this->blockLength / BUFFER_NUM;

    xGm.SetGlobalBuffer(
        (__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(),
        this->blockLength);
    yGm.SetGlobalBuffer(
        (__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(),
        this->blockLength);
    zGm.SetGlobalBuffer(
        (__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(),
        this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
  }

  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength],
                      this->tileLength);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileLength],
                      this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
    AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue<half>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
    AscendC::DataCopy(zGm[progress * this->tileLength], zLocal,
                      this->tileLength);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<half> xGm;
  AscendC::GlobalTensor<half> yGm;
  AscendC::GlobalTensor<half> zGm;

  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;

  nndeploy::op::AddCustomTilingData tiling_;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z, GM_ADDR tiling_gm) {
  KernelAdd op;
  op.Init(x, y, z, tiling_gm);
  op.Process();
}

namespace nndeploy {
namespace op {

void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
                   uint8_t *z, AddCustomTilingData *data) {
  // add_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}

}  // namespace op
}  // namespace nndeploy