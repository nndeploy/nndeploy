#include "nndeploy/op/op_reshape.h"

#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class AscendCLOpReshape : public OpReshape {
 public:
  AscendCLOpReshape() {}
  virtual ~AscendCLOpReshape() {}

  virtual base::Status init() {
    // 参数
    auto param = dynamic_cast<ir::ReshapeParam*>(op_desc_.op_param_.get());
    allowzero_ = static_cast<int64_t>(param->allowzero_);

    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() { 
    base::Status status = OpReshape::preRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("preRun failed.\n");
      return status;
    }
    return base::kStatusCodeOk;
  }
  // inferShape已经把事情做完了，这里只需要把输入数据拷贝到输出即可
  virtual base::Status run() {
    if (outputs_[0]->getData() != inputs_[0]->getData()) {
      size_t size_output = outputs_[0]->getSize();
      size_t size_input = inputs_[0]->getSize();
      size_t size_min = std::min(size_output, size_input);
      aclError ret = aclrtMemcpyAsync(
          outputs_[0]->getData(), size_min, inputs_[0]->getData(), size_min,
          ACL_MEMCPY_DEVICE_TO_DEVICE, inner_stream_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtMemcpyAsync failed.\n");
        return base::kStatusCodeErrorOpAscendCL;
      }
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() { 
    base::Status status = OpReshape::postRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("postRun failed.\n");
      return status;
    }
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Reshape";

  int allowzero_ = 0;

  aclrtStream inner_stream_ = nullptr;
  aclopAttr* attr_ = nullptr;
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeReshape, AscendCLOpReshape)

}  // namespace op
}  // namespace nndeploy
