#include "op_add_kernel.h"

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_NUM = 16;

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

template <typename x_T, typename y_T, typename z_T>
class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}

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
          (__gm__ x_T *)x + bodyLength * AscendC::GetBlockIdx(), bodyLength);
      yGm.SetGlobalBuffer(
          (__gm__ y_T *)y + bodyLength * AscendC::GetBlockIdx(), bodyLength);
      zGm.SetGlobalBuffer(
          (__gm__ z_T *)z + bodyLength * AscendC::GetBlockIdx(), bodyLength);
    } else {
      this->tileLength = tailLength;
      xGm.SetGlobalBuffer((__gm__ x_T *)x + bodyLength * bodyNum +
                              tailLength * (AscendC::GetBlockIdx() - bodyNum),
                          tailLength);
      yGm.SetGlobalBuffer((__gm__ y_T *)y + bodyLength * bodyNum +
                              tailLength * (AscendC::GetBlockIdx() - bodyNum),
                          tailLength);
      zGm.SetGlobalBuffer((__gm__ z_T *)z + bodyLength * bodyNum +
                              tailLength * (AscendC::GetBlockIdx() - bodyNum),
                          tailLength);
    }

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(x_T));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(y_T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(z_T));
  }

  __aicore__ inline void Process() {
    CopyIn();
    Compute();
    CopyOut();
  }

 private:
  __aicore__ inline void CopyIn() {
    AscendC::LocalTensor<x_T> xLocal = inQueueX.AllocTensor<x_T>();
    AscendC::LocalTensor<y_T> yLocal = inQueueY.AllocTensor<y_T>();
    AscendC::DataCopy(xLocal, xGm, this->tileLength);
    AscendC::DataCopy(yLocal, yGm, this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute() {
    AscendC::LocalTensor<x_T> xLocal = inQueueX.DeQue<x_T>();
    AscendC::LocalTensor<y_T> yLocal = inQueueY.DeQue<y_T>();
    AscendC::LocalTensor<z_T> zLocal = outQueueZ.AllocTensor<z_T>();
    AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue<z_T>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut() {
    AscendC::LocalTensor<z_T> zLocal = outQueueZ.DeQue<z_T>();
    AscendC::DataCopy(zGm, zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<x_T> xGm;
  AscendC::GlobalTensor<y_T> yGm;
  AscendC::GlobalTensor<z_T> zGm;

  uint32_t tileLength;
  nndeploy::op::AddCustomTilingData tiling_;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z, GM_ADDR tiling_gm) {
  KernelAdd<half, half, half> op;
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