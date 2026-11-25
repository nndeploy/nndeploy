/**
 * @brief LayerNorm 算子 Ascend C 实现
 *
 * 功能：对输入 tensor 在最后一个维度 H 上进行 LayerNorm 归一化
 * 公式：y = weight * (x - mean) / sqrt(variance + eps) + bias
 *
 * 输入 shape: [N, H] 或 [B, S, H]
 * 归一化维度：最后一个维度 H
 * 数据类型：float32
 * epsilon: 1e-5
 */

#include "op_layernorm_kernel.h"
#include "kernel_operator.h"

constexpr uint32_t LAYERNORM_FLOAT32 = 1;

// Buffer 配置：使用双缓冲提升性能
constexpr int32_t BUFFER_NUM = 2;

/**
 * @brief LayerNorm Kernel 实现类
 */
template <typename T>
class KernelLayerNorm {
 public:
  __aicore__ inline KernelLayerNorm() {}

  /**
   * @brief 初始化函数
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, LayerNormTilingData tiling_,
                              AscendC::TPipe* pipe_in) {
    pipe = pipe_in;
    tiling = tiling_;

    // 计算每个 block 处理的行数
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t rowStart = blockIdx * tiling.rowsPerCore;
    uint32_t rowEnd = (blockIdx + 1) * tiling.rowsPerCore;
    if (rowEnd > tiling.A1) {
      rowEnd = tiling.A1;
    }

    this->rowsThisCore = rowEnd - rowStart;
    this->R = tiling.R;
    this->rLengthAlign = tiling.rLengthAlign;
    this->invR = tiling.invR;

    // 设置全局 buffer
    inputGm.SetGlobalBuffer(
        (__gm__ T*)input + rowStart * rLengthAlign,
        rowsThisCore * rLengthAlign);
    weightGm.SetGlobalBuffer((__gm__ T*)weight, R);
    biasGm.SetGlobalBuffer((__gm__ T*)bias, R);
    outputGm.SetGlobalBuffer(
        (__gm__ T*)output + rowStart * rLengthAlign,
        rowsThisCore * rLengthAlign);

    // 计算每行需要的字节数
    uint32_t rowBytes = rLengthAlign * sizeof(T);

    // 初始化队列 Buffer
    // VECIN 队列用于输入
    pipe->InitBuffer(inQueueInput, BUFFER_NUM, rowBytes);
    pipe->InitBuffer(inQueueWeight, 1, R * sizeof(T));
    pipe->InitBuffer(inQueueBias, 1, R * sizeof(T));

    // VECOUT 队列用于输出和临时计算
    pipe->InitBuffer(outQueueOutput, BUFFER_NUM, rowBytes);
    pipe->InitBuffer(workQueue, 4, rowBytes);  // 4 个 buffer 用于中间计算
  }

  /**
   * @brief 主处理函数
   */
  __aicore__ inline void Process() {
    // Copy in weight and bias (每个 block 只需要一次)
    CopyInWeightBias();

    // 逐行处理
    for (uint32_t i = 0; i < rowsThisCore; i++) {
      CopyIn(i);
      Compute();
      CopyOut(i);
    }
  }

 private:
  /**
   * @brief 复制 weight 和 bias 到 UB
   */
  __aicore__ inline void CopyInWeightBias() {
    AscendC::LocalTensor<T> weightLocal = inQueueWeight.AllocTensor<T>();
    AscendC::LocalTensor<T> biasLocal = inQueueBias.AllocTensor<T>();

    AscendC::DataCopy(weightLocal, weightGm, R);
    AscendC::DataCopy(biasLocal, biasGm, R);

    inQueueWeight.EnQue(weightLocal);
    inQueueBias.EnQue(biasLocal);
  }

  /**
   * @brief 复制一行输入到 UB
   */
  __aicore__ inline void CopyIn(uint32_t rowIdx) {
    AscendC::LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
    uint32_t offset = rowIdx * rLengthAlign;

    // 使用 DataCopy 搬运一行数据
    if (rLengthAlign == R) {
      AscendC::DataCopy(inputLocal, inputGm[offset], R);
    } else {
      AscendC::DataCopy(inputLocal[0], inputGm[offset], R);
    }

    inQueueInput.EnQue(inputLocal);
  }

  /**
   * @brief 计算 LayerNorm
   *
   * 步骤：
   * 1. mean = sum(x) / R
   * 2. variance = sum((x - mean)^2) / R
   * 3. output = weight * (x - mean) / sqrt(variance + eps) + bias
   */
  __aicore__ inline void Compute() {
    // DeQue 输入
    AscendC::LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();

    // DeQue weight 和 bias
    AscendC::LocalTensor<T> weightLocal = inQueueWeight.DeQue<T>();
    AscendC::LocalTensor<T> biasLocal = inQueueBias.DeQue<T>();

    // 分配输出 tensor
    AscendC::LocalTensor<T> outputLocal = outQueueOutput.AllocTensor<T>();

    // 分配工作 buffer
    AscendC::LocalTensor<T> buf0 = workQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> buf1 = workQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> buf2 = workQueue.AllocTensor<T>();

    // eps 和 invR
    T eps = static_cast<T>(1e-5f);
    T invRVal = static_cast<T>(invR);

    // ========== Step 1: 计算 mean ==========
    // mean = sum(x) * invR
    // 先乘以 1/R 再求和，避免溢出
    AscendC::Muls<T>(buf0, inputLocal, invRVal, R);
    AscendC::PipeBarrier<PIPE_V>();

    // ReduceSum 计算 sum(x * invR) = mean
    // ReduceSum 输出在 buf0[0]
    AscendC::ReduceSum<T>(buf0, buf0, buf1, R);
    AscendC::PipeBarrier<PIPE_V>();

    // 将 mean 广播到整个 buffer (buf1)
    // 使用 Muls 将标量值复制到整个 buffer
    AscendC::Muls<T>(buf1, buf0[0], 1.0f, R);
    AscendC::PipeBarrier<PIPE_V>();

    // ========== Step 2: 计算 x - mean ==========
    AscendC::Sub<T>(buf2, inputLocal, buf1, R);
    AscendC::PipeBarrier<PIPE_V>();

    // ========== Step 3: 计算 variance ==========
    // (x - mean)^2
    AscendC::Mul<T>(buf0, buf2, buf2, R);
    AscendC::PipeBarrier<PIPE_V>();

    // * invR
    AscendC::Muls<T>(buf0, buf0, invRVal, R);
    AscendC::PipeBarrier<PIPE_V>();

    // ReduceSum 得到 variance
    AscendC::ReduceSum<T>(buf0, buf0, buf1, R);
    AscendC::PipeBarrier<PIPE_V>();

    // ========== Step 4: 计算 rsqrt(variance + eps) ==========
    // variance + eps
    AscendC::Adds<T>(buf0, buf0, eps, R);
    AscendC::PipeBarrier<PIPE_V>();

    // rsqrt
    AscendC::Rsqrt<T>(buf0, buf0, R);
    AscendC::PipeBarrier<PIPE_V>();

    // ========== Step 5: 计算 output = weight * (x-mean) * rsqrt + bias ==========
    // (x - mean) * rsqrt
    AscendC::Mul<T>(outputLocal, buf2, buf0, R);
    AscendC::PipeBarrier<PIPE_V>();

    // * weight
    AscendC::Mul<T>(outputLocal, outputLocal, weightLocal, R);
    AscendC::PipeBarrier<PIPE_V>();

    // + bias
    AscendC::Add<T>(outputLocal, outputLocal, biasLocal, R);
    AscendC::PipeBarrier<PIPE_V>();

    // EnQue 输出
    outQueueOutput.EnQue(outputLocal);

    // 释放工作 buffer
    workQueue.FreeTensor(buf0);
    workQueue.FreeTensor(buf1);
    workQueue.FreeTensor(buf2);

    // 释放输入和 weight/bias
    inQueueInput.FreeTensor(inputLocal);
    inQueueWeight.FreeTensor(weightLocal);
    inQueueBias.FreeTensor(biasLocal);
  }

  /**
   * @brief 复制一行输出到 GM
   */
  __aicore__ inline void CopyOut(uint32_t rowIdx) {
    AscendC::LocalTensor<T> outputLocal = outQueueOutput.DeQue<T>();
    uint32_t offset = rowIdx * rLengthAlign;

    if (rLengthAlign == R) {
      AscendC::DataCopy(outputGm[offset], outputLocal, R);
    } else {
      AscendC::DataCopy(outputGm[offset], outputLocal[0], R);
    }

    outQueueOutput.FreeTensor(outputLocal);
  }

 private:
  AscendC::TPipe* pipe;
  LayerNormTilingData tiling;

  // VECIN 队列用于输入
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueWeight;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueBias;

  // VECOUT 队列用于输出和工作 buffer
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 4> workQueue;

  // Global tensors
  AscendC::GlobalTensor<T> inputGm;
  AscendC::GlobalTensor<T> weightGm;
  AscendC::GlobalTensor<T> biasGm;
  AscendC::GlobalTensor<T> outputGm;

  // Tiling parameters
  uint32_t rowsThisCore;
  uint32_t R;
  uint32_t rLengthAlign;
  float invR;
};

/**
 * @brief 从 GM 复制 Tiling 数据到本地
 */
__aicore__ inline void CopyTilingData(LayerNormTilingData* dst, GM_ADDR src) {
  uint32_t* dstPtr = reinterpret_cast<uint32_t*>(dst);
  auto srcPtr = reinterpret_cast<__gm__ uint32_t*>(src);

  constexpr uint32_t tilingSize = sizeof(LayerNormTilingData) / sizeof(uint32_t);
  for (uint32_t i = 0; i < tilingSize; i++) {
    dstPtr[i] = srcPtr[i];
  }
}

/**
 * @brief LayerNorm Kernel 入口函数
 */
extern "C" __global__ __aicore__ void layernorm(
    GM_ADDR input, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR output, GM_ADDR tiling) {

  // 获取 Tiling 数据
  LayerNormTilingData tilingData;
  CopyTilingData(&tilingData, tiling);

  // 创建流水线
  AscendC::TPipe pipe;

  // 根据数据类型选择实现
  if (tilingData.dataType == LAYERNORM_FLOAT32) {
    KernelLayerNorm<float> op;
    op.Init(input, weight, bias, output, tilingData, &pipe);
    op.Process();
  } else {
    // 不支持的数据类型
    return;
  }
}
