#include "flag.h"
#include "nndeploy/device/buffer.h"
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
  if (FLAGS_tensor_pool_type ==
      "kTensorPool1DSharedObjectTypeGreedyByBreadth") {
    return net::kTensorPool1DSharedObjectTypeGreedyByBreadth;
  } else if (FLAGS_tensor_pool_type ==
             "kTensorPool1DSharedObjectTypeGreedyBySize") {
    return net::kTensorPool1DSharedObjectTypeGreedyBySize;
  } else if (FLAGS_tensor_pool_type ==
             "kTensorPool1DSharedObjectTypeGreedyBySizeImprove") {
    return net::kTensorPool1DSharedObjectTypeGreedyBySizeImprove;
  } else if (FLAGS_tensor_pool_type ==
             "kTensorPool1DOffsetCalculateTypeGreedyBySize") {
    return net::kTensorPool1DOffsetCalculateTypeGreedyBySize;
  } else if (FLAGS_tensor_pool_type ==
             "kTensorPool1DOffsetCalculateTypeGreedyByBreadth") {
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

  base::ModelType model_type = demo::getModelType();
  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();

  base::Status status = base::kStatusCodeOk;

  auto interpret_1 =
      std::shared_ptr<ir::Interpret>(ir::createInterpret(model_type));
  if (interpret_1 == nullptr) {
    NNDEPLOY_LOGE("ir::createInterpret failed.\n");
    return -1;
  }
  status = interpret_1->interpret(model_value);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret_1 failed\n");
    return -1;
  }

  auto interpret_2 =
      std::shared_ptr<ir::Interpret>(ir::createInterpret(model_type));
  if (interpret_2 == nullptr) {
    NNDEPLOY_LOGE("ir::createInterpret failed.\n");
    return -1;
  }
  std::vector<std::string> model_value_2;
  model_value_2.push_back(model_value[1]);
  status = interpret_2->interpret(model_value_2);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret_2 failed\n");
    return -1;
  }

  auto net_1 = std::make_shared<net::Net>();

  net_1->setInterpret(interpret_1.get());

  base::DeviceType device_type;
  // device_type.code_ = base::kDeviceTypeCodeCpu;
  device_type.code_ = base::kDeviceTypeCodeAscendCL;
  device_type.device_id_ = 0;
  net_1->setDeviceType(device_type);

  net::TensorPoolType tensor_pool_type = getTensorPoolType();
  net_1->setTensorPoolType(tensor_pool_type);
  bool is_external_tensor_pool_memory = true;
  net_1->setTensorPoolMemory(is_external_tensor_pool_memory);
  NNDEPLOY_LOGE("model path: %s\n", model_value[0].c_str());
  net_1->init();

  auto net_2 = std::make_shared<net::Net>();
  net_2->setInterpret(interpret_2.get());
  net_2->setDeviceType(device_type);
  net_2->setTensorPoolType(tensor_pool_type);
  net_2->setTensorPoolMemory(is_external_tensor_pool_memory);
  NNDEPLOY_LOGE("model path: %s\n", model_value[1].c_str());
  net_2->init();

  size_t memory_size_1 = net_1->getMemorySize();
  NNDEPLOY_LOGE("memory_size_1: %zu\n", memory_size_1);

  size_t memory_size_2 = net_2->getMemorySize();
  NNDEPLOY_LOGE("memory_size_2: %zu\n", memory_size_2);

  size_t max_memory_size = std::max(memory_size_1, memory_size_2);
  NNDEPLOY_LOGE("max_memory_size: %zu\n", max_memory_size);

  device::Device *device = device::getDevice(device_type);
  device::Buffer *buffer = new device::Buffer(device, max_memory_size);
  net_1->setMemory(buffer);
  net_2->setMemory(buffer);

  net_1->deinit();
  net_2->deinit();

  delete buffer;

  return 0;
}