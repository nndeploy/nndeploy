
#include "nndeploy/net/optimizer/fuse_conv_batchnorm.h"

namespace nndeploy {
namespace net {
FuseConvBatchNorm::FuseConvBatchNorm(){};
FuseConvBatchNorm::~FuseConvBatchNorm(){};

base::Status FuseConvBatchNorm::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  // 匹配 conv+batchnorm
  std::vector<ir::OpType> types = {ir::kOpTypeConv,
                                   ir::kOpTypeBatchNormalization};
  begin_op_index =
      seqPatternMatch(tensor_repository, op_repository, types, begin_op_index);
  if (begin_op_index == -1) {
    return base::kStatusCodeOk;
  }

  //  更新tensor_repository
  base::Status status = seqPatternMatchUpateTensorRepository(
      tensor_repository, op_repository, types, begin_op_index);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  // 更新Conv Op
  OpWrapper* first_op = op_repository[begin_op_index];
  OpWrapper* last_op = first_op->successors_[0];
  std::vector<device::Tensor*> outputs_tensors = last_op->op_->getAllOutput();
  first_op->op_->setAllOutput(
      outputs_tensors);  // 修改Conv的输出为BatchNorm的输出

  // Conv的顺序为： input、weight、bias（可能为空）
  // batchnorm的input顺序为 input、 scale、bias、mean、var
  device::Tensor* scale = last_op->op_->getInput(1);
  device::Tensor* bias = last_op->op_->getInput(2);
  device::Tensor* mean = last_op->op_->getInput(3);
  device::Tensor* var = last_op->op_->getInput(4);

  device::Tensor* conv_weight = first_op->op_->getInput(1);

  // TODO: 如果使用safetensor的权重加载方式，tensor的数据指针会mmap到文件上,
  // 导致权重不可修改； 如果修改会报段错误，使用 copy on write的方式进行修改
  auto previous_conv_weight = conv_weight;
  conv_weight = previous_conv_weight->clone();
  for (auto tensor_wrapper : tensor_repository) {
    if (tensor_wrapper->tensor_ == previous_conv_weight) {
      tensor_wrapper->tensor_ = conv_weight;
      break;
    }
  }
  delete previous_conv_weight;
  first_op->op_->setInput(conv_weight, 1);

  device::Tensor* conv_bias = nullptr;

  float* scale_data = reinterpret_cast<float*>(scale->getData());
  float* bias_data = reinterpret_cast<float*>(bias->getData());
  float* mean_data = reinterpret_cast<float*>(mean->getData());
  float* var_data = reinterpret_cast<float*>(var->getData());
  ir::BatchNormalizationParam* batchNormParam =
      (ir::BatchNormalizationParam*)last_op->op_->getParam().get();
  float epsilon = batchNormParam->epsilon_;

  int out_channels = conv_weight->getShape()[0];
  int in_channels = conv_weight->getShape()[1];
  int height = conv_weight->getShape()[2];
  int width = conv_weight->getShape()[3];

  // Conv有bias
  if (first_op->op_->getInput(2) != nullptr) {
    conv_bias = first_op->op_->getInput(2);
    auto previous_conv_bias = conv_bias;
    conv_bias = previous_conv_bias->clone();
    for (auto tensor_wrapper : tensor_repository) {
      if (tensor_wrapper->tensor_ == previous_conv_bias) {
        tensor_wrapper->tensor_ = conv_bias;
        break;
      }
    }
    delete previous_conv_bias;
    first_op->op_->setInput(conv_bias, 2);

  } else {  // Conv没有bias则创建一个bias
    device::TensorDesc conv_bias_desc(base::dataTypeOf<float>(),
                                      base::kDataFormatN, {out_channels});
    std::string name = first_op->name_ + ".bias";

    conv_bias =
        new device::Tensor(conv_weight->getDevice(), conv_bias_desc, name);
    conv_bias->set<float>(0);
  }

  for (int out_channel = 0; out_channel < out_channels; out_channel++) {
    float* conv_weight_data = reinterpret_cast<float*>(conv_weight->getData()) +
                              out_channel * in_channels * height * width;

    float var_sqrt = std::sqrt(var_data[out_channel] + epsilon);

    // 融合卷积权重
    for (int i = 0; i < in_channels * height * width; i++) {
      // std::cout << conv_weight_data[i] << std::endl;
      // conv_weight_data[i] = 0;
      // std::cout << conv_weight_data[i] << std::endl;
      conv_weight_data[i] =
          conv_weight_data[i] * scale_data[out_channel] / var_sqrt;
    }

    //融合进bias中

    float* conv_bias_data = reinterpret_cast<float*>(conv_bias->getData());
    conv_bias_data[out_channel] =
        (conv_bias_data[out_channel] - mean_data[out_channel]) *
            scale_data[out_channel] / var_sqrt +
        bias_data[out_channel];
  }

  // 删除原batchnorm的权重
  rmInputTensorAndMaybeDelete(last_op, tensor_repository);

  // 更新op_repository
  status = seqPatternMatchUpateOpRepository(tensor_repository, op_repository,
                                            types, begin_op_index);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  // for (auto op_wrapper : op_repository) {
  //   std::cout << op_wrapper->name_ << std::endl;
  //   std::cout << "produce" << std::endl;
  //   for (auto pre : op_wrapper->predecessors_) {
  //     std::cout << pre->name_ << std::endl;
  //   }
  //   for (auto con : op_wrapper->successors_) {
  //     std::cout << con->name_ << std::endl;
  //   }
  // }

  // for (auto tensor_wrapper : tensor_repository) {
  //   std::cout << tensor_wrapper->name_ << std::endl;
  //   std::cout << "produce" << std::endl;
  //   for (auto pre : tensor_wrapper->producers_) {
  //     std::cout << pre->name_ << std::endl;
  //   }
  //   for (auto con : tensor_wrapper->consumers_) {
  //     std::cout << con->name_ << std::endl;
  //   }
  // }

  return this->optimize(tensor_repository, op_repository, begin_op_index);
}

TypeOptPassRegister<TypeOptPassCreator<FuseConvBatchNorm>>
    g_fuse_conv_batchnorm_register(base::kDeviceTypeCodeCpu,
                                   kOptPassTypeFuseConvBatchNorm);

}  // namespace net
}  // namespace nndeploy