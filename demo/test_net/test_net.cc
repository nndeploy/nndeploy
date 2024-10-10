#include "nndeploy/interpret/interpret.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "test.h"

using namespace nndeploy;

class CannTest : public op::ModelDesc {
 public:
  CannTest() {};
  ~CannTest() {};
  void init() {
    auto input =
        op::makeInput(this, "input", base::dataTypeOf<float>(), {1, 3, 16, 16});
    // auto conv1 =
    //     makeConv(this, input, std::make_shared<ConvParam>(), "weight",
    //     "bias");
    // auto relu1 = makeRelu(this, conv1);
    auto softmax =
        op::makeSoftMax(this, input, std::make_shared<op::SoftmaxParam>());
    op::makeOutput(this, softmax);
  }
};

int main() {
  // net::TestNet testNet;
  // testNet.init();

  std::shared_ptr<interpret::OnnxInterpret> onnx_interpret =
      std::make_shared<interpret::OnnxInterpret>();
  std::vector<std::string> model_value;
  model_value.push_back("/root/model/yolov8n.onnx");
  // model_value.push_back("/root/model/modified_yolov8n.onnx");
  NNDEPLOY_LOGE("hello world\n");
  base::Status status = onnx_interpret->interpret(model_value);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("interpret failed\n");
    return -1;
  }
  NNDEPLOY_LOGE("hello world\n");
  op::ModelDesc *md = onnx_interpret->getModelDesc();
  if (md == nullptr) {
    NNDEPLOY_LOGE("get model desc failed\n");
    return -1;
  }
  NNDEPLOY_LOGE("hello world\n");

  // md->dump(std::cout);

  // NNDEPLOY_LOGE("hello world\n");
  // auto cann_model = std::make_shared<CannTest>();
  // cann_model->init();

  NNDEPLOY_LOGE("hello world\n");
  auto cann_net = std::make_shared<net::Net>();
  // cann_net->setModelDesc(cann_model.get());
  cann_net->setModelDesc(md);
  NNDEPLOY_LOGE("hello world\n");

  base::DeviceType device_type;
  device_type.code_ = base::kDeviceTypeCodeAscendCL;
  device_type.device_id_ = 0;
  cann_net->setDeviceType(device_type);

  cann_net->init();
  NNDEPLOY_LOGE("hello world\n");

  // cann_net->dump(std::cout);
  // NNDEPLOY_LOGE("hello world\n");

  cann_net->preRun();
  NNDEPLOY_LOGE("hello world\n");
  // cann_net->run();
  // NNDEPLOY_LOGE("hello world\n");
  // cann_net->postRun();
  // NNDEPLOY_LOGE("hello world\n");

  cann_net->deinit();
  NNDEPLOY_LOGE("hello world\n");

  return 0;
}
