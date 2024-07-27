#include "nndeploy/interpret/interpret.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "test.h"

using namespace nndeploy;

int main() {
  // net::TestNet testNet;
  // testNet.init();

  std::shared_ptr<interpret::OnnxInterpret> onnx_interpret =
      std::make_shared<interpret::OnnxInterpret>();
  std::vector<std::string> model_value;
  model_value.push_back(
      "/home/always/huggingface/nndeploy/model_zoo/detect/yolo/yolov8n.onnx");
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

  md->dump(std::cout);

  return 0;
}