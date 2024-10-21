#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/op_param.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/op.h"

using namespace nndeploy;

class TestDesc : public ir::ModelDesc {
 public:
  void init() {
    auto input =
        op::makeInput(this, "input", base::dataTypeOf<float>(), {1, 1, 8, 8});
    auto conv_param = std::make_shared<ir::ConvParam>();
    conv_param->kernel_shape_ = {3, 3};
    auto conv1 = op::makeConv(this, input, conv_param, "weight", "bias");
    auto relu1 = op::makeRelu(this, conv1);
    auto relu2 = op::makeRelu(this, relu1);
    // auto softmax = op::makeSoftMax(this, relu1, std::make_shared<ir::SoftmaxParam>());
    // op::makeOutput(this, softmax);
    // TODO：除最后一个节点外，中间节点的输出tensor不能为模型的输出节点（该要点还未实现,需完善OptPass::seqPatternMatch）
    // op::makeOutput(this, relu1);
    op::makeOutput(this, relu2);
  }
};

int main() {
  auto test_desc = new TestDesc();
  test_desc->init();

  base::DeviceType device_type = base::kDeviceTypeCodeCpu;
  device_type.device_id_ = 0;
  auto device = device::getDevice(device_type);

  device::TensorDesc weight_desc(base::dataTypeOf<float>(), base::kDataFormatOIHW, {32, 1, 3, 3});
  test_desc->weights_["weight"] = new device::Tensor(device, weight_desc, "weight");
  device::TensorDesc bias_desc(base::dataTypeOf<float>(), base::kDataFormatN, {32});
  test_desc->weights_["bias"] = new device::Tensor(device, bias_desc, "bias");
  for (auto x : test_desc->weights_) {
    std::cout << x.first << std::endl;
  }
  test_desc->dump(std::cout);
  // test_desc->weights_["weight"]->reshape({32,1,3,3});
  // test_desc->weights_["weight"]->setName("weight");
  // test_desc->weights_["bias"]->reshape({32});
  // test_desc->weights_["bias"]->setName("bias");
  auto test_net = std::make_shared<net::Net>();
  test_net->setModelDesc(test_desc);
  test_net->setDeviceType(device_type);

  NNDEPLOY_LOGE("init net\n");

  test_net->init();

  NNDEPLOY_LOGE("dump net\n");
  test_net->dump(std::cout);

  test_net->preRun();
  test_net->run();
  test_net->postRun();

  NNDEPLOY_LOGE("deinit net\n");
  test_net->deinit();

  NNDEPLOY_LOGE("delete test_desc\n");
  delete test_desc;

  return 0;
}
