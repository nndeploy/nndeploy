// #ifdef ENABLE_NNDEPLOY_OP_ASCEND_C
// #include <iostream>

// #include "acl/acl.h"
// #include "aclrtlaunch_add.h"
// #include "nndeploy/op/ascend_cl/op_convert.h"
// #include "nndeploy/op/ascend_cl/op_include.h"
// #include "nndeploy/op/ascend_cl/op_util.h"
// #include "nndeploy/op/op.h"
// #include "nndeploy/op/op_binary.h"
// #include "op_add_kernel.h"
// #include "tiling/platform/platform_ascendc.h"
// #include "tiling/tiling_api.h"

// void getTilingData(uint32_t input_size, uint8_t &block_num,
//                    nndeploy::op::AddTilingData &tiling_data) {
//   auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
//   auto aiv_num = platform->GetCoreNumAiv();

//   auto total_elements = input_size;

//   const uint32_t block_size = 32;
//   uint32_t align_num = block_size / sizeof(uint16_t);
//   uint32_t ub_block_num = 64;
//   uint32_t tile_num;
//   uint32_t total_elements_align =
//       (total_elements + align_num - 1) / align_num * align_num;
//   uint32_t total_block_num = total_elements_align / align_num;

//   if (total_elements_align <= ub_block_num * align_num) {
//     // 如果input可以放在一个ub里，使用单核
//     block_num = 1;
//   } else {
//     if (total_block_num % ub_block_num == 0) {
//       // 如果input可以被ub_block_num整除
//       if (total_block_num / ub_block_num <= aiv_num) {
//         block_num = total_block_num / ub_block_num;
//       } else {
//         block_num = aiv_num;
//       }
//     } else {
//       // 如果input不能被ub_block_num整除
//       if (total_block_num / ub_block_num + 1 <= aiv_num) {
//         block_num = total_block_num / ub_block_num + 1;
//       } else {
//         block_num = aiv_num;
//       }
//     }
//   }

//   // NNDEPLOY_LOGI("block_num: %d\n", block_num);

//   uint32_t block_length;
//   uint32_t tile_length;
//   uint32_t last_tile_length;
//   if (total_block_num % block_num == 0) {
//     // 如果input可以被核数整除
//     block_length = total_elements_align / block_num;
//     tile_num = block_length / align_num / ub_block_num;
//     if ((block_length / align_num) % ub_block_num == 0 || tile_num == 0) {
//       // 如果核内可以均分
//       if (tile_num == 0) {
//         tile_num = 1;
//       }
//       if (block_length < ub_block_num * align_num) {
//         tile_length = (block_length / align_num + 1) / 2 * 2 * align_num;
//         last_tile_length = tile_length;
//       } else {
//         tile_length = ub_block_num * align_num;
//         last_tile_length = tile_length;
//       }
//     } else {
//       // 如果核内不能均分
//       tile_num = tile_num + 1;
//       tile_length = ub_block_num * align_num;
//       last_tile_length = block_length - (tile_num - 1) * tile_length;
//     }
//     tiling_data.blockLength = block_length;
//     tiling_data.tileNum = tile_num;
//     tiling_data.tileLength = tile_length;
//     tiling_data.lastTileLength = last_tile_length;
//     tiling_data.tilingKey = 1;
//   } else {
//     // 如果input不能被核数整除
//     uint32_t head_num = total_block_num % block_num;
//     uint32_t tail_num = block_num - head_num;

//     // 计算大块和小块的数量
//     uint32_t head_block_length =
//         ((total_elements_align + block_num - 1) / block_num + align_num - 1)
//         / align_num * align_num;
//     uint32_t tail_block_length =
//         (total_elements_align / block_num / align_num) * align_num;

//     uint32_t head_tile_num = head_block_length / align_num / ub_block_num;
//     uint32_t head_tile_length;
//     uint32_t head_last_tile_length;
//     if ((head_block_length / align_num) % ub_block_num == 0 ||
//         head_tile_num == 0) {
//       // 如果核内可以均分
//       if (head_tile_num == 0) {
//         head_tile_num = 1;
//       }
//       if (head_block_length < ub_block_num * align_num) {
//         head_tile_length =
//             (head_block_length / align_num + 1) / 2 * 2 * align_num;
//         head_last_tile_length = head_tile_length;
//       } else {
//         head_tile_length = ub_block_num * align_num;
//         head_last_tile_length = head_tile_length;
//       }
//     } else {
//       // 如果核内不能均分
//       head_tile_num = head_tile_num + 1;
//       head_tile_length = ub_block_num * align_num;
//       head_last_tile_length =
//           head_block_length - (head_tile_num - 1) * head_tile_length;
//     }

//     uint32_t tail_tile_num = tail_block_length / align_num / ub_block_num;
//     uint32_t tail_tile_length;
//     uint32_t tail_last_tile_length;
//     if ((tail_block_length / align_num) % ub_block_num == 0 ||
//         tail_tile_num == 0) {
//       // 如果核内可以均分
//       if (tail_tile_num == 0) {
//         tail_tile_num = 1;
//       }
//       if (tail_block_length < ub_block_num * align_num) {
//         tail_tile_length =
//             (tail_block_length / align_num + 1) / 2 * 2 * align_num;
//         tail_last_tile_length = tail_tile_length;
//       } else {
//         tail_tile_length = ub_block_num * align_num;
//         tail_last_tile_length = tail_tile_length;
//       }
//     } else {
//       // 如果核内不能均分
//       tail_tile_num = tail_tile_num + 1;
//       tail_tile_length = ub_block_num * align_num;
//       tail_last_tile_length =
//           tail_block_length - (tail_tile_num - 1) * tail_tile_length;
//     }

//     tiling_data.headNum = head_num;
//     tiling_data.tailNum = tail_num;
//     tiling_data.headBlockLength = head_block_length;
//     tiling_data.tailBlockLength = tail_block_length;
//     tiling_data.headTileNum = head_tile_num;
//     tiling_data.tailTileNum = tail_tile_num;
//     tiling_data.headTileLength = head_tile_length;
//     tiling_data.tailTileLength = tail_tile_length;
//     tiling_data.headLastTileLength = head_last_tile_length;
//     tiling_data.tailLastTileLength = tail_last_tile_length;
//     tiling_data.tilingKey = 2;
//   }
//   return;
// }

// int main() {
//   CHECK_ACL_STATUS(aclInit(nullptr));
//   int32_t deviceId = 0;
//   CHECK_ACL_STATUS(aclrtSetDevice(deviceId));
//   aclrtStream stream = nullptr;
//   CHECK_ACL_STATUS(aclrtCreateStream(&stream));

//   // 输入长度
//   size_t inputByteSize = 256 * 256 * sizeof(uint16_t);
//   size_t outputByteSize = 256 * 256 * sizeof(uint16_t);

//   uint8_t *xHost, *yHost, *zHost;
//   uint8_t *xDevice, *yDevice, *zDevice;

//   CHECK_ACL_STATUS(aclrtMallocHost((void **)(&xHost), inputByteSize));
//   CHECK_ACL_STATUS(aclrtMallocHost((void **)(&yHost), inputByteSize));
//   CHECK_ACL_STATUS(aclrtMallocHost((void **)(&zHost), outputByteSize));

//   // 输入初始化
//   for (size_t i = 0; i < inputByteSize / sizeof(uint16_t); i++) {
//     xHost[i] = 1;
//     yHost[i] = 1;
//   }

//   CHECK_ACL_STATUS(
//       aclrtMalloc((void **)&xDevice, inputByteSize,
//       ACL_MEM_MALLOC_HUGE_FIRST));
//   CHECK_ACL_STATUS(
//       aclrtMalloc((void **)&yDevice, inputByteSize,
//       ACL_MEM_MALLOC_HUGE_FIRST));
//   CHECK_ACL_STATUS(aclrtMalloc((void **)&zDevice, outputByteSize,
//                                ACL_MEM_MALLOC_HUGE_FIRST));

//   CHECK_ACL_STATUS(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
//                                ACL_MEMCPY_HOST_TO_DEVICE));
//   CHECK_ACL_STATUS(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize,
//                                ACL_MEMCPY_HOST_TO_DEVICE));

//   AddTilingData tiling_data;
//   uint8_t block_num = 1;
//   getTilingData(inputByteSize / sizeof(uint16_t), block_num, tiling_data);
//   AddTilingData *buf = &tiling_data;
//   void *tiling_device = nullptr;
//   size_t tiling_size = sizeof(nndeploy::op::AddTilingData);
//   std::cout << "block_num: " << static_cast<int>(block_num) << std::endl;
//   CHECK_ACL_STATUS(aclrtMalloc((void **)&tiling_device, tiling_size,
//                                ACL_MEM_MALLOC_HUGE_FIRST));
//   CHECK_ACL_STATUS(aclrtMemcpyAsync(tiling_device, tiling_size, (void *)buf,
//                                     tiling_size, ACL_MEMCPY_HOST_TO_DEVICE,
//                                     stream));
//   CHECK_ACL_STATUS(aclrtSynchronizeStream(stream));

//   ACLRT_LAUNCH_KERNEL(add)
//   (block_num, stream, xDevice, yDevice, zDevice,
//    reinterpret_cast<uint8_t *>(tiling_device));
//   CHECK_ACL_STATUS(aclrtSynchronizeStream(stream));

//   // std::string error_msg = aclGetRecentErrMsg();
//   // std::cout << "error_msg: " << error_msg << std::endl;
//   CHECK_ACL_STATUS(aclrtMemcpy(zHost, outputByteSize, zDevice,
//   outputByteSize,
//                                ACL_MEMCPY_DEVICE_TO_HOST));

//   // 输出验证
//   int value = 0;
//   for (size_t i = 0; i < outputByteSize / sizeof(uint16_t); i++) {
//     if (zHost[i] != 2) {
//       value = 1;
//       std::cout << "Op add run failed!" << std::endl;
//       break;
//     }
//   }
//   if (value == 0) {
//     std::cout << "Op add run success!" << std::endl;
//   }

//   CHECK_ACL_STATUS(aclrtFree(xDevice));
//   CHECK_ACL_STATUS(aclrtFree(yDevice));
//   CHECK_ACL_STATUS(aclrtFree(zDevice));
//   CHECK_ACL_STATUS(aclrtFree(tiling_device));
//   CHECK_ACL_STATUS(aclrtFreeHost(xHost));
//   CHECK_ACL_STATUS(aclrtFreeHost(yHost));
//   CHECK_ACL_STATUS(aclrtFreeHost(zHost));

//   CHECK_ACL_STATUS(aclrtDestroyStream(stream));
//   CHECK_ACL_STATUS(aclrtResetDevice(deviceId));
//   CHECK_ACL_STATUS(aclFinalize());

//   return 0;
// }

// #endif

int main() { return 0; }
