#include "op_mul_kernel.h"

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

__aicore__ inline void CopyTiling(nndeploy::op::MulTilingData *tiling,
                                  GM_ADDR tiling_gm) {
  uint32_t *tiling_ptr = reinterpret_cast<uint32_t *>(tiling);
  __gm__ uint32_t *tiling_gm_ptr =
      reinterpret_cast<__gm__ uint32_t *>(tiling_gm);

  for (int i = 0; i < sizeof(nndeploy::op::MulTilingData) / sizeof(uint32_t);
       i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

template <typename x_T, typename y_T, typename z_T>
class KernelMul {
 public:
  __aicore__ inline KernelMul() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              GM_ADDR tiling_gm, AscendC::TPipe *pipe_in) {
    CopyTiling(&tiling_, tiling_gm);
    uint32_t tilingKey = tiling_.tilingKey;

    pipe = pipe_in;

    if (tilingKey == 1) {
      this->blockLength = tiling_.blockLength;
      this->tileNum = tiling_.tileNum;
      this->tileLength = tiling_.tileLength / BUFFER_NUM;
      this->lastTileLength = tiling_.lastTileLength;

      // AscendC::printf("this->blockLength: %d\n", this->blockLength);
      // AscendC::printf("this->tileNum: %d\n", this->tileNum);
      // AscendC::printf("this->tileLength: %d\n", this->tileLength);
      // AscendC::printf("this->lastTileLength: %d\n", this->lastTileLength);

      xGm.SetGlobalBuffer(
          (__gm__ x_T *)x + this->blockLength * AscendC::GetBlockIdx(),
          this->blockLength);
      yGm.SetGlobalBuffer(
          (__gm__ y_T *)y + this->blockLength * AscendC::GetBlockIdx(),
          this->blockLength);
      zGm.SetGlobalBuffer(
          (__gm__ z_T *)z + this->blockLength * AscendC::GetBlockIdx(),
          this->blockLength);
    }

    if (tilingKey == 2) {
      this->headNum = tiling_.headNum;
      this->tailNum = tiling_.tailNum;
      this->headLength = tiling_.headBlockLength;
      this->tailLength = tiling_.tailBlockLength;
      this->headTileNum = tiling_.headTileNum;
      this->tailTileNum = tiling_.tailTileNum;
      this->headTileLength = tiling_.headTileLength;
      this->tailTileLength = tiling_.tailTileLength;
      this->headLastTileLength = tiling_.headLastTileLength;
      this->tailLastTileLength = tiling_.tailLastTileLength;

      // AscendC::printf("this->headNum: %d\n", this->headNum);
      // AscendC::printf("this->tailNum: %d\n", this->tailNum);
      // AscendC::printf("this->headLength: %d\n", this->headLength);
      // AscendC::printf("this->tailLength: %d\n", this->tailLength);
      // AscendC::printf("this->headTileNum: %d\n", this->headTileNum);
      // AscendC::printf("this->tailTileNum: %d\n", this->tailTileNum);
      // AscendC::printf("this->headTileLength: %d\n", this->headTileLength);
      // AscendC::printf("this->tailTileLength: %d\n", this->tailTileLength);
      // AscendC::printf("this->headLastTileLength: %d\n",
      //                 this->headLastTileLength);
      // AscendC::printf("this->tailLastTileLength: %d\n",
      //                 this->tailLastTileLength);

      if (AscendC::GetBlockIdx() < this->headNum) {
        this->tileLength = this->headTileLength / BUFFER_NUM;
        this->lastTileLength = this->headLastTileLength;
        this->tileNum = this->headTileNum;
        xGm.SetGlobalBuffer(
            (__gm__ x_T *)x + this->headLength * AscendC::GetBlockIdx(),
            this->headLength);
        yGm.SetGlobalBuffer(
            (__gm__ y_T *)y + this->headLength * AscendC::GetBlockIdx(),
            this->headLength);
        zGm.SetGlobalBuffer(
            (__gm__ z_T *)z + this->headLength * AscendC::GetBlockIdx(),
            this->headLength);
      } else {
        this->tileLength = this->tailTileLength / BUFFER_NUM;
        this->lastTileLength = this->tailLastTileLength;
        this->tileNum = this->tailTileNum;
        xGm.SetGlobalBuffer(
            (__gm__ x_T *)x + this->headLength * this->headNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->headNum),
            this->tailLength);
        yGm.SetGlobalBuffer(
            (__gm__ y_T *)y + this->headLength * this->headNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->headNum),
            this->tailLength);
        zGm.SetGlobalBuffer(
            (__gm__ z_T *)z + this->headLength * this->headNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->headNum),
            this->tailLength);
      }
    }
    pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(x_T));
    pipe->InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(y_T));
    pipe->InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(z_T));
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
    // AscendC::LocalTensor<x_T> xLocal = inQueueX.AllocTensor<x_T>();
    // AscendC::LocalTensor<y_T> yLocal = inQueueY.AllocTensor<y_T>();
    // AscendC::DataCopy(xLocal, xGm, this->tileLength);
    // AscendC::DataCopy(yLocal, yGm, this->tileLength);
    // inQueueX.EnQue(xLocal);
    // inQueueY.EnQue(yLocal);
    AscendC::LocalTensor<x_T> xLocal = inQueueX.AllocTensor<x_T>();
    AscendC::LocalTensor<y_T> yLocal = inQueueY.AllocTensor<y_T>();

    if (BUFFER_NUM == 1) {
      if (progress == (this->tileNum - 1)) {
        if (progress == 0) {
          AscendC::DataCopy(xLocal, xGm, this->tileLength);
          AscendC::DataCopy(yLocal, yGm, this->tileLength);
        } else {
          AscendC::DataCopy(
              xLocal,
              xGm[(progress - 1) * this->tileLength + this->lastTileLength],
              this->tileLength);
          AscendC::DataCopy(
              yLocal,
              yGm[(progress - 1) * this->tileLength + this->lastTileLength],
              this->tileLength);
        }
      } else {
        AscendC::DataCopy(xLocal[0], xGm[progress * this->tileLength],
                          this->tileLength);
        AscendC::DataCopy(yLocal[0], yGm[progress * this->tileLength],
                          this->tileLength);
      }
    }

    if (BUFFER_NUM == 2) {
      if (progress == (this->tileNum * BUFFER_NUM - 2) ||
          progress == (this->tileNum * BUFFER_NUM - 1)) {
        AscendC::DataCopy(
            xLocal[0],
            xGm[(progress - 2) * this->tileLength + this->lastTileLength],
            this->tileLength);
        AscendC::DataCopy(
            yLocal[0],
            yGm[(progress - 2) * this->tileLength + this->lastTileLength],
            this->tileLength);
      } else {
        AscendC::DataCopy(xLocal[0], xGm[progress * this->tileLength],
                          this->tileLength);
        AscendC::DataCopy(yLocal[0], yGm[progress * this->tileLength],
                          this->tileLength);
      }
    }
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    // AscendC::LocalTensor<x_T> xLocal = inQueueX.DeQue<x_T>();
    // AscendC::LocalTensor<y_T> yLocal = inQueueY.DeQue<y_T>();
    // AscendC::LocalTensor<z_T> zLocal = outQueueZ.AllocTensor<z_T>();
    // AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
    // outQueueZ.EnQue<z_T>(zLocal);
    // inQueueX.FreeTensor(xLocal);
    // inQueueY.FreeTensor(yLocal);
    AscendC::LocalTensor<x_T> xLocal = inQueueX.DeQue<x_T>();
    AscendC::LocalTensor<y_T> yLocal = inQueueY.DeQue<y_T>();
    AscendC::LocalTensor<z_T> zLocal = outQueueZ.AllocTensor<z_T>();
    AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue<z_T>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    // AscendC::LocalTensor<z_T> zLocal = outQueueZ.DeQue<z_T>();
    // AscendC::DataCopy(zGm, zLocal, this->tileLength);
    // outQueueZ.FreeTensor(zLocal);
    AscendC::LocalTensor<z_T> zLocal = outQueueZ.DeQue<z_T>();
    if (BUFFER_NUM == 1) {
      if (progress == (this->tileNum - 1)) {
        if (progress == 0) {
          AscendC::DataCopy(zGm[0], zLocal[0], this->tileLength);
        } else {
          AscendC::DataCopy(
              zGm[(progress - 1) * this->tileLength + this->lastTileLength],
              zLocal[0], this->tileLength);
        }
      } else {
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal[0],
                          this->tileLength);
      }
    }

    if (BUFFER_NUM == 2) {
      if (progress == (this->tileNum * BUFFER_NUM - 2) ||
          progress == (this->tileNum * BUFFER_NUM - 1)) {
        AscendC::DataCopy(
            zGm[(progress - 2) * this->tileLength + this->lastTileLength],
            zLocal[0], this->tileLength);
      } else {
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal[0],
                          this->tileLength);
      }
    }
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe *pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<x_T> xGm;
  AscendC::GlobalTensor<y_T> yGm;
  AscendC::GlobalTensor<z_T> zGm;

  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  uint32_t lastTileLength;
  uint32_t headNum;
  uint32_t headLength;
  uint32_t headTileNum;
  uint32_t headTileLength;
  uint32_t headLastTileLength;
  uint32_t tailNum;
  uint32_t tailLength;
  uint32_t tailTileNum;
  uint32_t tailTileLength;
  uint32_t tailLastTileLength;

  nndeploy::op::MulTilingData tiling_;
};

extern "C" __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                          GM_ADDR tiling_gm) {
  AscendC::TPipe pipe;
  KernelMul<half, half, half> op;
  op.Init(x, y, z, tiling_gm, &pipe);
  op.Process();
}

// namespace nndeploy {
// namespace op {

// void mul_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y,
//                    uint8_t *z, MulTilingData *data) {
//   mul<<<blockDim, nullptr, stream>>>(x, y, z, data);
// }

// }  // namespace op
// }  // namespace nndeploy