#include "aclnnop/aclnn_sigmoid.h"
#include "nndeploy/op/ascend_cl/op_convert.h"
#include "nndeploy/op/ascend_cl/op_include.h"
#include "nndeploy/op/ascend_cl/op_util.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class AscendCLOpSigmoid : public OpUnary {
 public:
  AscendCLOpSigmoid() {}
  virtual ~AscendCLOpSigmoid() {}

  virtual base::Status init() {
    // 流
    device::Device* device = device::getDevice(device_type_);
    inner_stream_ = (aclrtStream)device->getCommandQueue();

    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status preRun() {
    // 输入输出
    inner_input_ =
        AscendCLOpConvert::convertFromTensor(inputs_[0], ACL_FORMAT_ND);
    inner_output_ =
        AscendCLOpConvert::convertFromTensor(outputs_[0], ACL_FORMAT_ND);

    // 创建算子
    aclnnStatus aclnn_status = aclnnSigmoidGetWorkspaceSize(
        inner_input_, inner_output_, &workspace_size_, &executor_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSigmoidGetWorkspaceSize failed.");
    return base::kStatusCodeOk;
  }
  virtual base::Status run() {
    // 输入输出
    aclnnStatus aclnn_status =
        aclnnSigmoid(workspace_, workspace_size_, executor_, inner_stream_);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
                                 base::kStatusCodeErrorOpAscendCL,
                                 "aclnnSigmoid failed.");

    // 同步流
#if 0
    std::string name_ = outputs_[0]->getName();
    if (name_ == "/model.1/act/Sigmoid_output_0") {
      aclrtSynchronizeStream(inner_stream_);
      std::string path = "./net_output/";
      std::string name = outputs_[0]->getName();
      std::string filename = name;
      size_t pos = 0;
      while ((pos = filename.find('/')) != std::string::npos) {
        filename.replace(pos, 1, "_");
      }
      filename = path + filename + "inner.csv";
      std::ofstream output_file(filename, std::ios::trunc);
      if (output_file.is_open()) {
        outputs_[0]->print(output_file);
        output_file.close();
      } else {
        NNDEPLOY_LOGE("无法打开文件：%s", filename.c_str());
      }
    }
#endif
    return base::kStatusCodeOk;
  }
  virtual base::Status postRun() {
    aclDestroyTensor(inner_input_);
    aclDestroyTensor(inner_output_);
    return base::kStatusCodeOk;
  }

 private:
  std::string inner_op_type_ = "Sigmoid";

  aclTensor* inner_input_ = nullptr;
  aclTensor* inner_output_ = nullptr;
  aclOpExecutor* executor_;

  aclrtStream inner_stream_;
  aclopAttr* attr_{nullptr};
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
                         ir::kOpTypeSigmoid, AscendCLOpSigmoid)

}  // namespace op
}  // namespace nndeploy
