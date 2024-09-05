// #include "nndeploy/op/op_maxpool.h"

// #include "aclnnop/aclnn_max_pool.h"
// #include "nndeploy/op/ascend_cl/acl_op_convert.h"
// #include "nndeploy/op/ascend_cl/acl_op_include.h"
// #include "nndeploy/op/ascend_cl/acl_op_util.h"
// #include "nndeploy/op/op.h"

// namespace nndeploy {
// namespace op {

// class AscendCLOpMaxPool : public OpMaxPool {
//  public:
// AscendCLOpMaxPool() {}
// virtual ~AscendCLOpMaxPool() {}

//   virtual base::Status init() {
//     // 参数
//     MaxPoolParam *param = (MaxPoolParam *)op_desc_.op_param_.get();
//     kernel_shape_ = AclOpConvert::convertFromIntVector(param->kernel_shape_);
//     stride_ = AclOpConvert::convertFromIntVector(param->strides_);
//     auto_pad_ = 0;
//     if (param->auto_pad_ == "NOTSET") {
//       auto_pad_ = 0;
//     } else {
//       NNDEPLOY_LOGE("Unsupported auto_pad: %s", param->auto_pad_.c_str());
//     }
//     padding_ = AclOpConvert::convertFromIntVector(param->pads_);
//     dilation_ = AclOpConvert::convertFromIntVector(param->dilations_);
//     ceil_mode_ = (int64_t)param->ceil_mode_;

//     // 流
//     device::Device *device = device::getDevice(device_type_);
//     inner_stream_ = (aclrtStream)device->getCommandQueue();

//     return base::kStatusCodeOk;
//   }
//   virtual base::Status deinit() {
//     aclDestroyIntArray(kernel_shape_);
//     aclDestroyIntArray(stride_);
//     aclDestroyIntArray(padding_);
//     aclDestroyIntArray(dilation_);
//     return base::kStatusCodeOk;
//   }
//   virtual base::Status preRun() {
//     // 输入输出
//     inner_input_ = AclOpConvert::convertFromTensor(inputs_[0]);
//     inner_output_ = AclOpConvert::convertFromTensor(outputs_[0]);

//     // 创建算子
//     aclnnStatus aclnn_status = aclnnMaxPoolGetWorkspaceSize(
//         inner_input_, kernel_shape_, stride_, auto_pad_, padding_, dilation_,
//         ceil_mode_, inner_output_, &workspace_size_, &executor_);
//     NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
//                                  base::kStatusCodeErrorOpAscendCL,
//                                  "aclnnMaxPoolGetWorkspaceSize failed.");
//     return base::kStatusCodeOk;
//   }
//   virtual base::Status run() {
//     aclnnStatus aclnn_status =
//         aclnnMaxPool(workspace_, workspace_size_, executor_, inner_stream_);
//     NNDEPLOY_RETURN_VALUE_ON_NEQ(aclnn_status, ACL_SUCCESS,
//                                  base::kStatusCodeErrorOpAscendCL,
//                                  "aclnnSoftmax failed.");

//     return base::kStatusCodeOk;
//   }
//   virtual base::Status postRun() {
//     aclDestroyTensor(inner_input_);
//     aclDestroyTensor(inner_output_);
//     // aclDestroyExecutor(executor_);
//     return base::kStatusCodeOk;
//   }

//  private:
//   std::string inner_op_type_ = "MaxPool";

//   aclTensor *inner_input_;
//   aclIntArray *kernel_shape_;
//   aclIntArray *stride_;
//   int64_t auto_pad_;
//   aclIntArray *padding_;
//   aclIntArray *dilation_;
//   int64_t ceil_mode_;
//   aclTensor *inner_output_;

//   aclOpExecutor *executor_;

//   aclrtStream inner_stream_;
//   aclopAttr *attr_{nullptr};
// };

// REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
//                          kOpTypeMaxPool, AscendCLOpMaxPool)

// }  // namespace op
// }  // namespace nndeploy
