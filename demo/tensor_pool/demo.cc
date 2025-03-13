#include "flag.h"
#include "nndeploy/framework.h"
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/op.h"

using namespace nndeploy;

DEFINE_string(tensor_pool_type, "", "tensor pool type");

net::TensorPoolType getTensorPoolType() {
  if (FLAGS_tensor_pool_type == "kTensorPool1DSharedObjectTypeGreedyByBreadth") {
    return net::kTensorPool1DSharedObjectTypeGreedyByBreadth;
  } else if (FLAGS_tensor_pool_type == "kTensorPool1DSharedObjectTypeGreedyBySize") {
    return net::kTensorPool1DSharedObjectTypeGreedyBySize;
  } else if (FLAGS_tensor_pool_type == "kTensorPool1DSharedObjectTypeGreedyBySizeImprove") {
    return net::kTensorPool1DSharedObjectTypeGreedyBySizeImprove;
  } else if (FLAGS_tensor_pool_type == "kTensorPool1DOffsetCalculateTypeGreedyBySize") {
    return net::kTensorPool1DOffsetCalculateTypeGreedyBySize;
  } else if (FLAGS_tensor_pool_type == "kTensorPool1DOffsetCalculateTypeGreedyByBreadth") {
    return net::kTensorPool1DOffsetCalculateTypeGreedyByBreadth;
  } else if (FLAGS_tensor_pool_type == "kTensorPool1DNone") {
    return net::kTensorPool1DNone;
  }
  return net::kTensorPool1DSharedObjectTypeGreedyBySize;  // 默认使用正确的方法
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  base::ModelType model_type = demo::getModelType();
  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();

  base::Status status = base::kStatusCodeOk;

  auto interpret = std::shared_ptr<ir::Interpret>(ir::createInterpret(model_type));
  if (interpret == nullptr) {
    NNDEPLOY_LOGE("ir::createInterpret failed.\n");
    return -1;
  }
  status = interpret->interpret(model_value);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret failed\n");
    return -1;
  }

  auto net = std::make_shared<net::Net>();
  net->setInterpret(interpret.get());

  base::DeviceType device_type;
  // device_type.code_ = base::kDeviceTypeCodeCpu;
  device_type.code_ = base::kDeviceTypeCodeAscendCL;
  device_type.device_id_ = 0;
  net->setDeviceType(device_type);

  net::TensorPoolType tensor_pool_type = getTensorPoolType();
  net->setTensorPoolType(tensor_pool_type);

  net->init();

  net->dump(std::cout);

  net->deinit();

  // ret = nndeployFrameworkDeinit();
  // if (ret != 0) {
  //   NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
  //   return ret;
  // }
  return 0;
}