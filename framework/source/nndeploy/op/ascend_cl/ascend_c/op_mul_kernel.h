#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_MUL_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_MUL_KERNEL_H_

namespace nndeploy {
namespace op {

struct MulTilingData {
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  uint32_t lastTileLength;

  uint32_t headNum;
  uint32_t tailNum;

  uint32_t headBlockLength;
  uint32_t tailBlockLength;

  uint32_t headTileNum;
  uint32_t tailTileNum;

  uint32_t headTileLength;
  uint32_t tailTileLength;

  uint32_t headLastTileLength;
  uint32_t tailLastTileLength;

  uint32_t tilingKey;
};

}  // namespace op
}  // namespace nndeploy

#endif