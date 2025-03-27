#include "flag.h"
#include "nndeploy/framework.h"
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/op.h"

using namespace nndeploy;

DEFINE_string(model_json, "", "test.json");
DEFINE_string(model_safetensors, "", "test.safetensors");

std::string getModelJson() { return FLAGS_model_json; }

std::string getModelSafetensors() { return FLAGS_model_safetensors; }

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

  // 模型类型，例如:
  // kModelTypeOnnx/kModelTypeMnn/...
  base::ModelType model_type = demo::getModelType();
  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();
  std::string model_json = getModelJson();
  std::string model_safetensors = getModelSafetensors();

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
  status = interpret->saveModelToFile(model_json, model_safetensors);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("saveModelToFile failed\n");
    return -1;
  }

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}
