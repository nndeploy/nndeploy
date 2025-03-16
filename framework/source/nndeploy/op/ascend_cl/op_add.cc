#include "aclnnop/aclnn_add.h"
#include "ascend_c/op_add_kernel.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_binary.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace nndeploy {
namespace op {

#ifdef ENABLE_NNDEPLOY_OP_ASCEND_C
#include "acl/acl.h"
#include "aclrtlaunch_add.h"
class AscendCLOpAdd : public OpBinary {
 public:
  AscendCLOpAdd() {}
  virtual ~AscendCLOpAdd() {}

  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    // inner stream
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      inputs_0_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                     inputs_[0]->getName());
      inputs_[0]->copyTo(inputs_0_);
    } else {
      inputs_0_ = inputs_[0];
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      inputs_1_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                     inputs_[1]->getName());
      inputs_[1]->copyTo(inputs_1_);
    } else {
      inputs_1_ = inputs_[1];
    }

    // get input shape
    base::IntVector input0_shape = inputs_0_->getShape();
    base::IntVector input1_shape = inputs_1_->getShape();
    // check input shape
    if (input0_shape.size() != input1_shape.size()) {
      NNDEPLOY_LOGE(
          "Input tensors do not have the same number of dimensions.\n");
      return base::kStatusCodeErrorInvalidParam;
    }

    for (size_t i = 0; i < input1_shape.size(); ++i) {
      if (input0_shape[i] != input1_shape[i]) {
        NNDEPLOY_LOGE("Input tensors do not have the same shape.\n");
        return base::kStatusCodeErrorInvalidParam;
      }
    }

    getTilingData();
    AddTilingData* buf = &add_tiling_data_;
    size_t tiling_size = sizeof(AddTilingData);
    CHECK_ACL_STATUS(aclrtMalloc((void**)&tiling_device, tiling_size,
                                 ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_STATUS(aclrtMemcpyAsync(tiling_device, tiling_size, (void*)buf,
                                      tiling_size, ACL_MEMCPY_HOST_TO_DEVICE,
                                      inner_stream_));
    CHECK_ACL_STATUS(aclrtSynchronizeStream(inner_stream_));

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (inputs_0_ != nullptr) {
      if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
        delete inputs_0_;
        inputs_0_ = nullptr;
      }
    }
    if (inputs_1_ != nullptr) {
      if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
        delete inputs_1_;
        inputs_1_ = nullptr;
      }
    }
    if (tiling_device != nullptr) {
      aclrtFree(tiling_device);
      tiling_device = nullptr;
    }
    return Op::deinit();
  }
  virtual base::Status run() {
    uint8_t* input_data_0 = (uint8_t*)(inputs_0_->getData());
    uint8_t* input_data_1 = (uint8_t*)(inputs_1_->getData());
    uint8_t* output_data = (uint8_t*)(outputs_[0]->getData());

    ACLRT_LAUNCH_KERNEL(add)
    (block_num_, inner_stream_, input_data_0, input_data_1, output_data,
     reinterpret_cast<uint8_t*>(tiling_device));
    CHECK_ACL_STATUS(aclrtSynchronizeStream(inner_stream_));

    return base::kStatusCodeOk;
  }

  base::Status getTilingData() {
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto aiv_num = platform->GetCoreNumAiv();

    // platform profiling data
    // auto version = platform->GetSocVersion();
    // auto core_num = platform->GetCoreNum();
    // auto aic_num = platform->GetCoreNumAic();
    // uint64_t L0A, L0B, L0C, L1, L2, UB, HBM;
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, L0A);
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, L0B);
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, L0C);
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, L1);
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, L2);
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, UB);
    // platform->GetCoreMemSize(platform_ascendc::CoreMemType::HBM, HBM);

    // NNDEPLOY_LOGI("core_num: %d\n", core_num);
    // NNDEPLOY_LOGI("aic_num: %d\n", aic_num);
    // NNDEPLOY_LOGI("L0A: %lu\n", L0A);
    // NNDEPLOY_LOGI("L0B: %lu\n", L0B);
    // NNDEPLOY_LOGI("L0C: %lu\n", L0C);
    // NNDEPLOY_LOGI("L1: %lu\n", L1);
    // NNDEPLOY_LOGI("L2: %lu\n", L2);
    // NNDEPLOY_LOGI("UB: %lu\n", UB);
    // NNDEPLOY_LOGI("HBM: %lu\n", HBM);

    base::IntVector input_shape = inputs_0_->getShape();
    auto total_elements = std::accumulate(
        input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

    const uint32_t block_size = 32;
    uint32_t align_num = block_size / sizeof(uint16_t);
    uint32_t ub_block_num = 256;
    uint32_t tile_num;
    uint32_t total_elements_align =
        (total_elements + align_num - 1) / align_num * align_num;
    uint32_t total_block_num = total_elements_align / align_num;

    if (total_elements_align <= ub_block_num * align_num) {
      // 如果input可以放在一个ub里，使用单核
      block_num_ = 1;
    } else {
      if (total_block_num % ub_block_num == 0) {
        // 如果input可以被ub_block_num整除
        if (total_block_num / ub_block_num <= aiv_num) {
          block_num_ = total_block_num / ub_block_num;
        } else {
          block_num_ = aiv_num;
        }
      } else {
        // 如果input不能被ub_block_num整除
        if (total_block_num / ub_block_num + 1 <= aiv_num) {
          block_num_ = total_block_num / ub_block_num + 1;
        } else {
          block_num_ = aiv_num;
        }
      }
    }

    // NNDEPLOY_LOGI("block_num_: %d\n", block_num_);

    uint32_t block_length;
    uint32_t tile_length;
    uint32_t last_tile_length;
    if (total_block_num % block_num_ == 0) {
      // 如果input可以被核数整除
      block_length = total_elements_align / block_num_;
      tile_num = block_length / align_num / ub_block_num;
      if ((block_length / align_num) % ub_block_num == 0 || tile_num == 0) {
        // 如果核内可以均分
        if (tile_num == 0) {
          tile_num = 1;
        }
        if (block_length < ub_block_num * align_num) {
          tile_length = (block_length / align_num + 1) / 2 * 2 * align_num;
          last_tile_length = tile_length;
        } else {
          tile_length = ub_block_num * align_num;
          last_tile_length = tile_length;
        }
      } else {
        // 如果核内不能均分
        tile_num = tile_num + 1;
        tile_length = ub_block_num * align_num;
        last_tile_length = block_length - (tile_num - 1) * tile_length;
      }
      add_tiling_data_.blockLength = block_length;
      add_tiling_data_.tileNum = tile_num;
      add_tiling_data_.tileLength = tile_length;
      add_tiling_data_.lastTileLength = last_tile_length;
      add_tiling_data_.tilingKey = 1;
    } else {
      // 如果input不能被核数整除
      uint32_t head_num = total_block_num % block_num_;
      uint32_t tail_num = block_num_ - head_num;

      // 计算大块和小块的数量
      uint32_t head_block_length =
          ((total_elements_align + block_num_ - 1) / block_num_ + align_num -
           1) /
          align_num * align_num;
      uint32_t tail_block_length =
          (total_elements_align / block_num_ / align_num) * align_num;

      uint32_t head_tile_num = head_block_length / align_num / ub_block_num;
      uint32_t head_tile_length;
      uint32_t head_last_tile_length;
      if ((head_block_length / align_num) % ub_block_num == 0 ||
          head_tile_num == 0) {
        // 如果核内可以均分
        if (head_tile_num == 0) {
          head_tile_num = 1;
        }
        if (head_block_length < ub_block_num * align_num) {
          head_tile_length =
              (head_block_length / align_num + 1) / 2 * 2 * align_num;
          head_last_tile_length = head_tile_length;
        } else {
          head_tile_length = ub_block_num * align_num;
          head_last_tile_length = head_tile_length;
        }
      } else {
        // 如果核内不能均分
        head_tile_num = head_tile_num + 1;
        head_tile_length = ub_block_num * align_num;
        head_last_tile_length =
            head_block_length - (head_tile_num - 1) * head_tile_length;
      }

      uint32_t tail_tile_num = tail_block_length / align_num / ub_block_num;
      uint32_t tail_tile_length;
      uint32_t tail_last_tile_length;
      if ((tail_block_length / align_num) % ub_block_num == 0 ||
          tail_tile_num == 0) {
        // 如果核内可以均分
        if (tail_tile_num == 0) {
          tail_tile_num = 1;
        }
        if (tail_block_length < ub_block_num * align_num) {
          tail_tile_length =
              (tail_block_length / align_num + 1) / 2 * 2 * align_num;
          tail_last_tile_length = tail_tile_length;
        } else {
          tail_tile_length = ub_block_num * align_num;
          tail_last_tile_length = tail_tile_length;
        }
      } else {
        // 如果核内不能均分
        tail_tile_num = tail_tile_num + 1;
        tail_tile_length = ub_block_num * align_num;
        tail_last_tile_length =
            tail_block_length - (tail_tile_num - 1) * tail_tile_length;
      }

      add_tiling_data_.headNum = head_num;
      add_tiling_data_.tailNum = tail_num;
      add_tiling_data_.headBlockLength = head_block_length;
      add_tiling_data_.tailBlockLength = tail_block_length;
      add_tiling_data_.headTileNum = head_tile_num;
      add_tiling_data_.tailTileNum = tail_tile_num;
      add_tiling_data_.headTileLength = head_tile_length;
      add_tiling_data_.tailTileLength = tail_tile_length;
      add_tiling_data_.headLastTileLength = head_last_tile_length;
      add_tiling_data_.tailLastTileLength = tail_last_tile_length;
      add_tiling_data_.tilingKey = 2;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Add";

  device::Tensor* inputs_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;

  aclrtStream inner_stream_ = nullptr;

  void* tiling_device = nullptr;
  AddTilingData add_tiling_data_;

  uint8_t block_num_ = 0;
};
#else
class AscendCLOpAdd : public OpBinary {
 public:
  AscendCLOpAdd() {}
  virtual ~AscendCLOpAdd() {}

  virtual base::Status init() {
    base::Status status = Op::init();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ =
        (aclrtStream)stream_->as<device::AscendCLStream>()->getStream();

    if (device::isHostDeviceType(inputs_[0]->getDeviceType())) {
      inputs_0_ = new device::Tensor(device, inputs_[0]->getDesc(),
                                     inputs_[0]->getName());
      inputs_[0]->copyTo(inputs_0_);
      inner_input_0_ =
          AscendCLOpConvert::convertFromTensor(inputs_0_, ACL_FORMAT_ND);
    }
    if (device::isHostDeviceType(inputs_[1]->getDeviceType())) {
      inputs_1_ = new device::Tensor(device, inputs_[1]->getDesc(),
                                     inputs_[1]->getName());
      inputs_[1]->copyTo(inputs_1_);
      inner_input_1_ =
          AscendCLOpConvert::convertFromTensor(inputs_1_, ACL_FORMAT_ND);
    }

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    if (alpha_ != nullptr) {
      aclDestroyScalar(alpha_);
      alpha_ = nullptr;
    }
    if (inputs_0_ != nullptr) {
      if (inner_input_0_ != nullptr) {
        aclDestroyTensor(inner_input_0_);
        inner_input_0_ = nullptr;
      }
      delete inputs_0_;
      inputs_0_ = nullptr;
    }
    if (inputs_1_ != nullptr) {
      if (inner_input_1_ != nullptr) {
        aclDestroyTensor(inner_input_1_);
        inner_input_1_ = nullptr;
      }
      delete inputs_1_;
      inputs_1_ = nullptr;
    }
    return Op::deinit();
  }
  virtual base::Status preRun() {
    // 输入输出
    if (inner_input_0_ == nullptr) {
      inner_input_0_ =
          AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    }
    if (inner_input_1_ == nullptr) {
      inner_input_1_ =
          AscendCLOpConvert::convertFromTensor(inputs_[1], ACL_FORMAT_ND);
    }
    if (alpha_ == nullptr) {
      alpha_ =
          AscendCLOpConvert::convertFromScalar(1.0f, inputs_[0]->getDataType());
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(alpha_, "aclCreateScalar failed.");
    }
    if (inner_output_ == nullptr) {
      inner_output_ =
          AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);
    }

    // 创建算子
    if (executor_ == nullptr) {
      aclnnStatus aclnn_status =
          aclnnAddGetWorkspaceSize(inner_input_0_, inner_input_1_, alpha_,
                                   inner_output_, &workspace_size_, &executor_);
      if (aclnn_status != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclnnAddGetWorkspaceSize failed, error code: %d.\n",
                      aclnn_status);
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnAdd(workspace_, workspace_size_, executor_, inner_stream_);
    if (aclnn_status != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclnnAdd failed, error code: %d.\n", aclnn_status);
      return base::kStatusCodeErrorOpAscendCL;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    if (inputs_0_ == nullptr && inner_input_0_ != nullptr) {
      aclDestroyTensor(inner_input_0_);
      inner_input_0_ = nullptr;
    }
    if (inputs_1_ == nullptr && inner_input_1_ != nullptr) {
      aclDestroyTensor(inner_input_1_);
      inner_input_1_ = nullptr;
    }
    if (inner_output_ != nullptr) {
      aclDestroyTensor(inner_output_);
      inner_output_ = nullptr;
    }
    if (executor_ != nullptr) {
      executor_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Add";

  device::Tensor* inputs_0_ = nullptr;
  aclTensor* inner_input_0_ = nullptr;
  device::Tensor* inputs_1_ = nullptr;
  aclTensor* inner_input_1_ = nullptr;
  aclScalar* alpha_ = nullptr;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_ = nullptr;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};
#endif

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeAscendCL, ir::kOpTypeAdd, AscendCLOpAdd)

}  // namespace op
}  // namespace nndeploy
