// #include "nndeploy/op/op_transpose.h"

// #include "aclnnop/aclnn_transpose.h"
// #include "nndeploy/op/ascend_cl/acl_op_convert.h"
// #include "nndeploy/op/ascend_cl/acl_op_include.h"
// #include "nndeploy/op/ascend_cl/acl_op_util.h"
// #include "nndeploy/op/op.h"

// namespace nndeploy {
// namespace op {

// class AscendCLOpTranspose : public OpTranspose {
//  public:
//   AscendCLOpTranspose() {}
//   virtual ~AscendCLOpTranspose() {}

//   virtual base::Status init() {
//     // 参数
//     auto param = dynamic_cast<TransposeParam*>(op_desc_.op_param_.get());
//     mode_ = static_cast<int64_t>(param->mode_);

//     // 流
//     device::Device* device = device::getDevice(device_type_);
//     inner_stream_ = (aclrtStream)device->getCommandQueue();

//     return base::kStatusCodeOk;
//   }
//   virtual base::Status deinit() {
//     if (scales_ != nullptr) {
//       aclDestoryFloatArray(scales_);
//     }
//     return base::kStatusCodeOk;
//   }
//   virtual base::Status preRun() {
//     // 输入输出
//     inner_input_ = AclOpConvert::convertFromTensor(inputs_[0]);
//     base::DataType data_type = inputs_[1]->getDataType();
//     if (data_type.code_ != base::kDataTypeCodeFp) {
//       NNDEPLOY_LOG_ERROR("Transpose only support float data type.");
//       return base::kStatusCodeErrorInvalidParam;
//     }
//     if (scales_ == nullptr) {
//       float* data = (float)inputs_[1]->getData();
//       size_t size = inputs_[1]->getSize() >> 2;
//       scales_ = aclCreateFloatArray(data, size);
//     }
//     inner_output_ = AclOpConvert::convertFromTensor(outputs_[0]);

//     // 创建算子
//     char* mode = mode_.c_str();
//     aclnnStatus aclnn_status = aclnnTransposeGetWorkspaceSize(
//         inner_inputs_, scales_, mode, inner_output_, &workspace_size_,
//         &executor_);
//     NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
//                                  base::kStatusCodeErrorOpAscendCL,
//                                  "aclnnTransposeGetWorkspaceSize failed.");
//     return base::kStatusCodeOk;
//   }
//   virtual base::Status run() {
//     // 输入输出
//     aclnnStatus aclnn_status =
//         aclnnTranspose(workspace_, workspace_size_, executor_,
//         inner_stream_);
//     NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
//                                  base::kStatusCodeErrorOpAscendCL,
//                                  "aclnnCat failed.");

//     return base::kStatusCodeOk;
//   }
//   virtual base::Status postRun() {
//     aclDestroyTensorList(inner_inputs_);
//     aclDestroyTensor(inner_output_);
//     // aclDestroyExecutor(executor_);
//     return base::kStatusCodeOk;
//   }

//  private:
//   // TODO: 待完善
//   std::string inner_op_type_ = "Transpose";

//   aclTensor* inner_input_ = nullptr;
//   aclFloatArray* scales_ = nullptr;
//   std::string mode_ = "nearest";
//   aclTensor* inner_output_ = nullptr;
//   aclOpExecutor* executor_;

//   aclrtStream inner_stream_;
//   aclopAttr* attr_{nullptr};
// };

// REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
//                          kOpTypeTranspose, AscendCLOpTranspose)

// }  // namespace op
// }  // namespace nndeploy
