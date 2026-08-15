#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_LAYERNORM_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_LAYERNORM_KERNEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LayerNorm Tiling 数据结构
 *
 * Host 端填充，传递给 kernel 使用
 */
struct LayerNormTilingData {
  uint32_t A1;                  // 外层元素总数：N (2D) 或 B*S (3D)
  uint32_t R;                   // 归一化维度大小：H
  uint32_t rowsPerCore;         // 每个 core 处理的行数
  uint32_t usedCoreNum;         // 实际使用的 core 数
  uint32_t rLengthAlign;        // R 方向对齐后的长度（32 字节对齐）
  float invR;                   // 1.0 / R，用于计算均值
  uint32_t tmpBufSize;          // 临时缓冲区大小
  uint32_t dataType;            // 数据类型：0=FP16, 1=FP32
};

#ifdef __cplusplus
}
#endif

#endif
