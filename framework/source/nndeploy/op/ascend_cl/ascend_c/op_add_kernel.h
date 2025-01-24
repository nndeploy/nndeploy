#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_ADD_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_ADD_KERNEL_H_

struct AddCustomTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

#endif