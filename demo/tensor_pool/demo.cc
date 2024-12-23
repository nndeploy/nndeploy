// #include <experimental/filesystem>
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
// #include "nndeploy/ir/onnx/onnx_interpret.h"
#include "nndeploy/framework.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/op.h"
#include "test.h"

using namespace nndeploy;

class CannTest : public ir::ModelDesc {
 public:
  CannTest(){};
  ~CannTest(){};
  void init() {
    auto input =
        op::makeInput(this, "input", base::dataTypeOf<float>(), {1, 1, 8, 8});
    // auto conv1 =
    //     makeConv(this, input, std::make_shared<ConvParam>(), "weight",
    //     "bias");
    // auto relu1 = makeRelu(this, conv1);
    auto softmax_0 =
        op::makeSoftMax(this, input, std::make_shared<ir::SoftmaxParam>());
    auto softmax_1 =
        op::makeSoftMax(this, input, std::make_shared<ir::SoftmaxParam>());

    auto add = op::makeAdd(this, softmax_0, softmax_1);

    op::makeOutput(this, add);
  }
};

int main() {
  // net::TestNet testNet;
  // testNet.init();

  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  auto onnx_interpret =
      std::shared_ptr<ir::Interpret>(ir::createInterpret(base::kModelTypeOnnx));
  std::vector<std::string> model_value;
  // model_value.push_back("D:\\github\\nndeploy\\build\\yolov8n.onnx");
  model_value.push_back("yolov8n.onnx");

  base::Status status = onnx_interpret->interpret(model_value);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret failed\n");
    return -1;
  }

  // onnx_interpret->dump(std::cout);
  onnx_interpret->saveModelToFile("yolov8n.json", "yolov8n.safetensors");


  auto default_interpret = std::shared_ptr<ir::Interpret>(
      ir::createInterpret(base::kModelTypeDefault));
  std::vector<std::string> new_model_value;
  new_model_value.push_back("yolov8n.json");
  new_model_value.push_back("yolov8n.safetensors");
  base::Status status;
  status = default_interpret->interpret(new_model_value);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret failed\n");
    return -1;
  }
  default_interpret->saveModelToFile("yolov8n_test.json",
                                     "yolov8n_test.safetensors");

  // ir::ModelDesc *md = onnx_interpret->getModelDesc();
  ir::ModelDesc *md = default_interpret->getModelDesc();
  if (md == nullptr) {
    NNDEPLOY_LOGE("get model desc failed\n");
    return -1;
  }

  // md->dump(std::cout);

  // auto md = new CannTest();
  // md->init();
  auto cann_net = std::make_shared<net::Net>();
  // cann_net->setModelDesc(cann_model.get());
  cann_net->setModelDesc(md);

  base::DeviceType device_type;
  device_type.code_ = base::kDeviceTypeCodeCpu;
  device_type.device_id_ = 0;
  cann_net->setDeviceType(device_type);

  cann_net->init();

  // cann_net->dump(std::cout);

  std::vector<device::Tensor *> inputs = cann_net->getAllInput();
  inputs[0]->set<float>(1.0f);
  // inputs[0]->print();

  cann_net->preRun();

  cann_net->run();

  cann_net->postRun();

  // std::vector<device::Tensor *>inputs = cann_net->getAllInput();

  std::vector<device::Tensor *> outputs = cann_net->getAllOutput();
  // outputs[0]->print();

  cann_net->deinit();

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}
