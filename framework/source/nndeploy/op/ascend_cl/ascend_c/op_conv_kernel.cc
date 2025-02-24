#include "op_conv_kernel.h"

#include "kernel_operator.h"

constexpr int32_t ALIGN_NUM = 16;

__aicore__ inline void CopyTiling(nndeploy::op::Conv2dTilingData *tiling,
                                  GM_ADDR tiling_gm) {
  uint32_t *tiling_ptr = reinterpret_cast<uint32_t *>(tiling);
  __gm__ uint32_t *tiling_gm_ptr =
      reinterpret_cast<__gm__ uint32_t *>(tiling_gm);

  for (int i = 0; i < sizeof(nndeploy::op::Conv2dTilingData) / sizeof(uint32_t);
       i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

template <typename dst_T, typename fm_T, typename we_T>
class KernelConv2d {
 public:
  __aicore__ inline KernelConv2d() {}
  __aicore__ inline void Init(GM_ADDR fmGm, GM_ADDR weGm, GM_ADDR dstGm,
                              GM_ADDR tiling_gm, AscendC::TPipe *tmpPipe) {
    CopyTiling(&tiling_data, tiling_gm);
    pipe = tmpPipe;
    batchSize = tiling_data.batchSize;
    inHeight = tiling_data.inHeight;
    inWidth = tiling_data.inWidth;
    outHeight = tiling_data.outHeight;
    outWidth = tiling_data.outHeight;
    inChannel = tiling_data.inChannel;
    outChannel = tiling_data.outChannel;
    kernelSize = tiling_data.kernelSize;
    stride = tiling_data.stride;
    dilation = tiling_data.dilation;
    padding = tiling_data.padding;
    coreNum = tiling_data.coreNum;

    inChannelAlign = (inChannel + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    outChannelAlign = (outChannel + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    fmWidthBlock =
        (alignBlock / inChannelAlign - (kernelSize - stride)) / stride;
    fmWidthRounds = outWidth / fmWidthBlock;
    fmWidthTail = outWidth % fmWidthBlock;

    fmBlockLen = inChannel * sizeof(fm_T);
    fmPadWidth = inChannelAlign - inChannel;
    fmBlockCount = fmWidthBlock * stride + (kernelSize - stride);
    fmTailBlockCount = fmWidthTail * stride + (kernelSize - stride);

    weBlockLen = inChannel * sizeof(we_T);
    weBlockCount = kernelSize * kernelSize * outChannel;
    wePadWidth = inChannelAlign - inChannel;

    dstBlockCount = fmWidthBlock;
    dstTailBlockCount = fmWidthTail;

    // AscendC::printf("fmWidthBlock=%d fmWidthRounds=%d fmWidthTail=%d\n",
    //                 fmWidthBlock, fmWidthRounds, fmWidthTail);

    taskNum = batchSize * outHeight;
    taskNumPerCore = AscendC::DivCeil(taskNum, coreNum);

    curBlockIdx = AscendC::GetBlockIdx();
    startOffset = curBlockIdx * taskNumPerCore;
    endOffset = (curBlockIdx + 1) * taskNumPerCore;
    if (endOffset > taskNum) {
      endOffset = taskNum;
    }

    fmGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fm_T *>(fmGm),
                             batchSize * inHeight * inWidth * inChannel);
    weGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ we_T *>(weGm),
                             outChannel * kernelSize * kernelSize * inChannel);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dst_T *>(dstGm),
                              batchSize * outChannel * outHeight * outWidth);

    // AscendC::printf("alignBlock=%d kernelSize=%d inChannelAlign=%d\n",
    //                 alignBlock, kernelSize, inChannelAlign);

    pipe->InitBuffer(fmBatchUb1, alignBlock * sizeof(fm_T));
    pipe->InitBuffer(fmBatchUb2, alignBlock * sizeof(fm_T));
    pipe->InitBuffer(fmBatchUb3, alignBlock * sizeof(fm_T));
    pipe->InitBuffer(fmBatchUb4, kernelSize * inChannelAlign * sizeof(fm_T));
    pipe->InitBuffer(weBatchUb1, outChannel * kernelSize * kernelSize *
                                     inChannelAlign * sizeof(we_T));
    pipe->InitBuffer(dstBatchUb1, outChannelAlign * inChannelAlign *
                                      kernelSize * sizeof(dst_T));
    pipe->InitBuffer(dstBatchUb2,
                     fmWidthBlock * outChannelAlign * sizeof(dst_T));
  }
  __aicore__ inline void Process() { ComputeNH(); }

 private:
  __aicore__ inline void TailPart() {}
  __aicore__ inline void MainPart() {
    AscendC::DataCopyExtParams weCopyParams{weBlockCount, weBlockLen, 0, 0, 0};
    AscendC::DataCopyPadExtParams<we_T> wePadParams{true, 0, fmPadWidth, 0};
    AscendC::DataCopyPad(weBatchLt1, weGlobal, weCopyParams,
                         wePadParams);  // 从GM->VECCALC搬运data
    // AscendC::DumpTensor(weBatchLt1, 1, 128);

    for (uint32_t idx1 = 0; idx1 < fmWidthRounds; idx1++) {
      inOffset = baseOffset + (idx1 * fmWidthBlock) * stride;
      AscendC::DataCopyExtParams fmCopyParams{fmBlockCount, fmBlockLen, 0, 0,
                                              0};
      AscendC::DataCopyPadExtParams<fm_T> fmPadParams{true, 0, fmPadWidth, 0};
      AscendC::DataCopyPad(fmBatchLt1, fmGlobal[inOffset * inChannel],
                           fmCopyParams,
                           fmPadParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(
          fmBatchLt2, fmGlobal[(inOffset + inWidth) * inChannel], fmCopyParams,
          fmPadParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(fmBatchLt3,
                           fmGlobal[(inOffset + inWidth * 2) * inChannel],
                           fmCopyParams,
                           fmPadParams);  // 从GM->VECCALC搬运data
      pipe_barrier(PIPE_ALL);

      for (uint32_t idx2 = 0; idx2 < fmWidthBlock; idx2++) {
        for (uint32_t idx3 = 0; idx3 < outChannel; idx3++) {
          AscendC::Mul(
              dstBatchLt1[idx3 * kernelSize * inChannelAlign],
              fmBatchLt1[idx2 * stride * inChannelAlign],
              weBatchLt1[idx3 * kernelSize * kernelSize * inChannelAlign],
              kernelSize * inChannelAlign);
          AscendC::Mul(
              fmBatchLt4, fmBatchLt2[idx2 * stride * inChannelAlign],
              weBatchLt1[idx3 * kernelSize * kernelSize * inChannelAlign +
                         kernelSize * inChannelAlign],
              kernelSize * inChannelAlign);
          AscendC::Add(dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       fmBatchLt4, kernelSize * inChannelAlign);
          AscendC::Mul(
              fmBatchLt4, fmBatchLt3[idx2 * stride * inChannelAlign],
              weBatchLt1[idx3 * kernelSize * kernelSize * inChannelAlign +
                         kernelSize * 2 * inChannelAlign],
              kernelSize * inChannelAlign);
          AscendC::Add(dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       fmBatchLt4, kernelSize * inChannelAlign);
        }
        int32_t srcStride = kernelSize * inChannelAlign / ALIGN_NUM;
        AscendC::WholeReduceSum<dst_T, true>(
            dstBatchLt2[idx2 * outChannelAlign], dstBatchLt1,
            kernelSize * inChannelAlign, outChannel, 1, 1, srcStride);
      }
      pipe_barrier(PIPE_ALL);

      AscendC::DataCopyExtParams copyOutParams{dstBlockCount, outChannel * 2, 0,
                                               0, 0};
      AscendC::DataCopyPad(
          dstGlobal[outOffset + (idx1 * fmWidthBlock) * outChannel],
          dstBatchLt2, copyOutParams);
    }

    if (fmWidthTail > 0) {
      inOffset = baseOffset + (fmWidthRounds * fmWidthBlock) * stride;
      AscendC::DataCopyExtParams fmCopyParams{fmTailBlockCount, fmBlockLen, 0,
                                              0, 0};
      AscendC::DataCopyPadExtParams<fm_T> fmPadParams{true, 0, fmPadWidth, 0};
      AscendC::DataCopyPad(fmBatchLt1, fmGlobal[inOffset * inChannel],
                           fmCopyParams,
                           fmPadParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(
          fmBatchLt2, fmGlobal[(inOffset + inWidth) * inChannel], fmCopyParams,
          fmPadParams);  // 从GM->VECCALC搬运data
      AscendC::DataCopyPad(fmBatchLt3,
                           fmGlobal[(inOffset + inWidth * 2) * inChannel],
                           fmCopyParams,
                           fmPadParams);  // 从GM->VECCALC搬运data
      pipe_barrier(PIPE_ALL);
      // AscendC::DumpTensor(fmBatchLt1, 2, 128);
      // AscendC::DumpTensor(fmBatchLt2, 3, 128);
      // AscendC::DumpTensor(fmBatchLt3, 4, 128);

      for (uint32_t idx2 = 0; idx2 < fmWidthTail; idx2++) {
        for (uint32_t idx3 = 0; idx3 < outChannel; idx3++) {
          AscendC::Mul(
              dstBatchLt1[idx3 * kernelSize * inChannelAlign],
              fmBatchLt1[idx2 * stride * inChannelAlign],
              weBatchLt1[idx3 * kernelSize * kernelSize * inChannelAlign],
              kernelSize * inChannelAlign);
          AscendC::Mul(
              fmBatchLt4, fmBatchLt2[idx2 * stride * inChannelAlign],
              weBatchLt1[idx3 * kernelSize * kernelSize * inChannelAlign +
                         kernelSize * inChannelAlign],
              kernelSize * inChannelAlign);
          AscendC::Add(dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       fmBatchLt4, kernelSize * inChannelAlign);
          AscendC::Mul(
              fmBatchLt4, fmBatchLt3[idx2 * stride * inChannelAlign],
              weBatchLt1[idx3 * kernelSize * kernelSize * inChannelAlign +
                         kernelSize * 2 * inChannelAlign],
              kernelSize * inChannelAlign);
          AscendC::Add(dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       dstBatchLt1[idx3 * kernelSize * inChannelAlign],
                       fmBatchLt4, kernelSize * inChannelAlign);
        }
        int32_t srcStride = kernelSize * inChannelAlign / ALIGN_NUM;
        AscendC::WholeReduceSum<dst_T, true>(
            dstBatchLt2[idx2 * outChannelAlign], dstBatchLt1,
            kernelSize * inChannelAlign, outChannel, 1, 1, srcStride);
        // AscendC::printf("This idx index is %d\n", idx2);
        // AscendC::DumpTensor(dstBatchLt1, 0, 1024);
        // AscendC::DumpTensor(dstBatchLt2, 1, 256);
      }
      pipe_barrier(PIPE_ALL);

      AscendC::DataCopyExtParams copyOutParams{dstTailBlockCount,
                                               outChannel * 2, 0, 0, 0};
      AscendC::DataCopyPad(
          dstGlobal[outOffset + (fmWidthRounds * fmWidthBlock) * outChannel],
          dstBatchLt2, copyOutParams);
    }
  }
  __aicore__ inline void ComputeNH() {
    fmBatchLt1 = fmBatchUb1.Get<fm_T>();
    fmBatchLt2 = fmBatchUb2.Get<fm_T>();
    fmBatchLt3 = fmBatchUb3.Get<fm_T>();
    fmBatchLt4 = fmBatchUb4.Get<fm_T>();

    weBatchLt1 = weBatchUb1.Get<we_T>();

    dstBatchLt1 = dstBatchUb1.Get<dst_T>();
    dstBatchLt2 = dstBatchUb2.Get<dst_T>();

    for (uint32_t i = startOffset; i < endOffset; i++) {
      hOffset = i % outHeight;
      bOffset = i / outHeight;

      outOffset = i * outWidth * outChannel;

      inHeightOffset = hOffset * stride;
      baseOffset = (bOffset * inHeight + inHeightOffset) * inWidth;

      if (inHeightOffset + kernelSize > inHeight) {
        TailPart();
      } else {
        MainPart();
      }
    }
    // AscendC::DumpTensor(dstGlobal, 1, 256);
  }

 private:
  uint16_t batchSize;
  uint16_t inHeight;
  uint16_t inWidth;
  uint16_t outHeight;
  uint16_t outWidth;
  uint8_t kernelSize;
  uint16_t inChannel;
  uint32_t outChannel;
  uint16_t stride;
  uint8_t dilation;
  uint16_t padding;
  nndeploy::op::Conv2dTilingData tiling_data;

  uint16_t outBlockCount;
  uint32_t inChannelAlign;
  uint32_t outChannelAlign;

  uint32_t fmWidthBlock;
  uint32_t alignBlock = 64 * 64;
  uint32_t fmWidthRounds;
  uint32_t fmWidthTail;

  uint8_t fmPadWidth;
  uint16_t fmBlockLen;
  uint16_t fmBlockCount;
  uint16_t fmTailBlockCount;

  uint8_t wePadWidth;
  uint16_t weBlockLen;
  uint16_t weBlockCount;

  uint16_t dstBlockCount;
  uint16_t dstTailBlockCount;

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

  AscendC::TPipe *pipe;

  AscendC::GlobalTensor<fm_T> fmGlobal;
  AscendC::GlobalTensor<we_T> weGlobal;
  AscendC::GlobalTensor<dst_T> dstGlobal;

  AscendC::TBuf<AscendC::TPosition::VECCALC> fmBatchUb1, fmBatchUb2, fmBatchUb3,
      fmBatchUb4;
  AscendC::TBuf<AscendC::TPosition::VECCALC> weBatchUb1;
  AscendC::TBuf<AscendC::TPosition::VECCALC> dstBatchUb1, dstBatchUb2;
  AscendC::LocalTensor<fm_T> fmBatchLt1, fmBatchLt2, fmBatchLt3, fmBatchLt4;
  AscendC::LocalTensor<we_T> weBatchLt1;
  AscendC::LocalTensor<dst_T> dstBatchLt1, dstBatchLt2;
};

extern "C" __global__ __aicore__ void conv2d(GM_ADDR fmGm, GM_ADDR weGm,
                                             GM_ADDR dstGm, GM_ADDR tiling) {
  AscendC::TPipe pipe;
  KernelConv2d<half, half, half> op;
  op.Init(fmGm, weGm, dstGm, tiling, &pipe);
  op.Process();
}