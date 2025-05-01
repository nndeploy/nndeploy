#include "op_maxpool_kernel.h"

#include "kernel_operator.h"

constexpr uint32_t MAXPOOL_FLOAT16 = 0;
constexpr uint32_t MAXPOOL_FLOAT32 = 1;

template <typename x_T, typename y_T>
class KernelMaxPool2d {
 public:
  __aicore__ inline KernelMaxPool2d() {}
  __aicore__ inline void Init(GM_ADDR x_trans, GM_ADDR y_trans,
                              MaxPool2dTilingData tiling_data,
                              AscendC::TPipe *tmpPipe) {
    pipe = tmpPipe;
    batchSize = tiling_data.batchSize;
    channel = tiling_data.channel;
    inHeight = tiling_data.inHeight;
    inWidth = tiling_data.inWidth;
    outHeight = tiling_data.outHeight;
    outWidth = tiling_data.outWidth;
    coreNum = tiling_data.coreNum;
    stride = tiling_data.stride;
    kernelSize = tiling_data.kernelSize;

    taskNum = batchSize * outHeight;
    taskNumPerCore = AscendC::DivCeil(taskNum, coreNum);

    curBlockIdx = AscendC::GetBlockIdx();
    startOffset = curBlockIdx * taskNumPerCore;
    endOffset = (curBlockIdx + 1) * taskNumPerCore;
    if (endOffset > taskNum) {
      endOffset = taskNum;
    }

    int32_t align_num = 32 / sizeof(x_T);

    inChannelAlign = (channel + align_num - 1) / align_num * align_num;
    widthBlock = (alignBlock / inChannelAlign - (kernelSize - stride)) / stride;
    widthRounds = outWidth / widthBlock;
    widthTail = outWidth % widthBlock;

    blockLen = channel * sizeof(x_T);
    padWidth = inChannelAlign - channel;
    blockCount = widthBlock * stride + (kernelSize - stride);
    outBlockCount = widthBlock;
    tailBlockCount = widthTail * stride + (kernelSize - stride);
    tailOutBlockCount = widthTail;

    xTransGm.SetGlobalBuffer(reinterpret_cast<__gm__ x_T *>(x_trans),
                             batchSize * inHeight * inWidth * channel);
    yTransGm.SetGlobalBuffer(reinterpret_cast<__gm__ y_T *>(y_trans),
                             batchSize * outHeight * outWidth * channel);

    pipe->InitBuffer(xBatchUb1, alignBlock * sizeof(x_T));
    pipe->InitBuffer(xBatchUb2, alignBlock * sizeof(x_T));
    pipe->InitBuffer(xBatchUb3, alignBlock * sizeof(x_T));
    pipe->InitBuffer(xBatchUb4, alignBlock * sizeof(x_T));
  }
  __aicore__ inline void Process() { ComputeNH(); }

 private:
  __aicore__ inline void MainPart() {
    for (uint32_t idx1 = 0; idx1 < widthRounds; idx1++) {
      inOffset = baseOffset + (idx1 * widthBlock) * stride;
      AscendC::DataCopyExtParams copyParams{blockCount, blockLen, 0, 0, 0};
      AscendC::DataCopyPadExtParams<x_T> padParams{true, 0, padWidth, 0};
      AscendC::DataCopyPad(xBatchLt1, xTransGm[inOffset * channel], copyParams,
                           padParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(xBatchLt2, xTransGm[(inOffset + inWidth) * channel],
                           copyParams,
                           padParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(
          xBatchLt3, xTransGm[(inOffset + inWidth * 2) * channel], copyParams,
          padParams);  // 从GM->VECCALC搬运data

      pipe_barrier(PIPE_ALL);

      AscendC::Max(
          xBatchLt1, xBatchLt1, xBatchLt2,
          (widthBlock * stride + (kernelSize - stride)) * inChannelAlign);
      AscendC::Max(
          xBatchLt1, xBatchLt1, xBatchLt3,
          (widthBlock * stride + (kernelSize - stride)) * inChannelAlign);

      for (uint32_t idx2 = 0; idx2 < widthBlock; idx2++) {
        AscendC::Max(xBatchLt4[idx2 * inChannelAlign],
                     xBatchLt1[idx2 * stride * inChannelAlign],
                     xBatchLt1[idx2 * stride * inChannelAlign + inChannelAlign],
                     inChannelAlign);
        AscendC::Max(
            xBatchLt4[idx2 * inChannelAlign], xBatchLt4[idx2 * inChannelAlign],
            xBatchLt1[idx2 * stride * inChannelAlign + inChannelAlign * 2],
            inChannelAlign);
      }
      pipe_barrier(PIPE_ALL);

      AscendC::DataCopyExtParams copyOutParams{
          outBlockCount, static_cast<uint32_t>(channel * sizeof(x_T)), 0, 0, 0};
      AscendC::DataCopyPad(yTransGm[outOffset + (idx1 * widthBlock) * channel],
                           xBatchLt4, copyOutParams);
    }

    if (widthTail > 0) {
      inOffset = baseOffset + (widthRounds * widthBlock) * stride;
      AscendC::DataCopyExtParams copyParams{tailBlockCount, blockLen, 0, 0, 0};
      AscendC::DataCopyPadExtParams<x_T> padParams{true, 0, padWidth, 0};
      AscendC::DataCopyPad(xBatchLt1, xTransGm[inOffset * channel], copyParams,
                           padParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(xBatchLt2, xTransGm[(inOffset + inWidth) * channel],
                           copyParams,
                           padParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(
          xBatchLt3, xTransGm[(inOffset + inWidth * 2) * channel], copyParams,
          padParams);  // 从GM->VECCALC搬运data

      // AscendC::DumpTensor(xBatchLt1, 1, 32);
      // AscendC::DumpTensor(xBatchLt2, 2, 32);
      // AscendC::DumpTensor(xBatchLt3, 3, 32);

      pipe_barrier(PIPE_ALL);

      AscendC::Max(
          xBatchLt1, xBatchLt1, xBatchLt2,
          (widthTail * stride + (kernelSize - stride)) * inChannelAlign);
      AscendC::Max(
          xBatchLt1, xBatchLt1, xBatchLt3,
          (widthTail * stride + (kernelSize - stride)) * inChannelAlign);

      for (uint32_t idx2 = 0; idx2 < widthTail; idx2++) {
        AscendC::Max(xBatchLt4[idx2 * inChannelAlign],
                     xBatchLt1[idx2 * stride * inChannelAlign],
                     xBatchLt1[idx2 * stride * inChannelAlign + inChannelAlign],
                     inChannelAlign);
        AscendC::Max(
            xBatchLt4[idx2 * inChannelAlign], xBatchLt4[idx2 * inChannelAlign],
            xBatchLt1[idx2 * stride * inChannelAlign + inChannelAlign * 2],
            inChannelAlign);
      }

      pipe_barrier(PIPE_ALL);

      AscendC::DataCopyExtParams copyOutParams{
          tailOutBlockCount, static_cast<uint32_t>(channel * sizeof(x_T)), 0, 0,
          0};
      AscendC::DataCopyPad(
          yTransGm[outOffset + (widthRounds * widthBlock) * channel], xBatchLt4,
          copyOutParams);
    }
  }
  __aicore__ inline void TailPart() {}
  __aicore__ inline void ComputeNH() {
    xBatchLt1 = xBatchUb1.Get<x_T>();
    xBatchLt2 = xBatchUb2.Get<x_T>();
    xBatchLt3 = xBatchUb3.Get<x_T>();
    xBatchLt4 = xBatchUb4.Get<x_T>();
    for (uint32_t i = startOffset; i < endOffset; i++) {
      hOffset = i % outHeight;
      bOffset = i / outHeight;

      outOffset = i * outWidth * channel;

      inHeightOffset = hOffset * stride;
      baseOffset = (bOffset * inHeight + inHeightOffset) * inWidth;

      if (inHeightOffset + kernelSize > inHeight) {
        TailPart();
      } else {
        MainPart();
      }
    }
  }

 private:
  AscendC::TPipe *pipe;

  uint32_t batchSize;
  uint32_t channel;
  uint32_t inHeight;
  uint32_t inWidth;
  uint32_t outHeight;
  uint32_t outWidth;
  uint32_t kernelSize;
  uint32_t stride;
  uint32_t inChannelAlign;
  uint32_t inWidthAlign;
  uint32_t oneChannelSize;

  uint8_t padWidth;
  uint16_t blockLen;
  uint16_t blockCount;
  uint16_t outBlockCount;
  uint16_t tailBlockCount;
  uint16_t tailOutBlockCount;

  uint32_t widthBlock;
  uint32_t alignBlock = 64 * 64;
  uint32_t widthRounds;
  uint32_t widthTail;

  uint32_t coreNum;
  uint32_t taskNum;
  uint32_t taskNumPerCore;
  uint32_t curBlockIdx;
  uint32_t startOffset;
  uint32_t endOffset;
  uint32_t inOffset;

  uint32_t hOffset;
  uint32_t bOffset;
  uint32_t inHeightOffset;
  uint32_t baseOffset;
  uint32_t outOffset;

  AscendC::GlobalTensor<x_T> xTransGm;
  AscendC::GlobalTensor<y_T> yTransGm;
  AscendC::TBuf<AscendC::TPosition::VECCALC> xBatchUb1, xBatchUb2, xBatchUb3,
      xBatchUb4;
  AscendC::LocalTensor<x_T> xBatchLt1, xBatchLt2, xBatchLt3, xBatchLt4;
};

extern "C" __global__ __aicore__ void max_pool2d(GM_ADDR x_trans,
                                                 GM_ADDR y_trans,
                                                 MaxPool2dTilingData tiling) {
  if (tiling.dataType == MAXPOOL_FLOAT16) {
    AscendC::TPipe pipe;
    KernelMaxPool2d<half, half> op;
    op.Init(x_trans, y_trans, tiling, &pipe);
    op.Process();
  } else if (tiling.dataType == MAXPOOL_FLOAT32) {
    AscendC::TPipe pipe;
    KernelMaxPool2d<float, float> op;
    op.Init(x_trans, y_trans, tiling, &pipe);
    op.Process();
  } else {
    return;
  }
}