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

    op::makeOutput(this, relu1);
  }
};

int main() {
  auto test_net = std::make_shared<net::Net>();
  auto test_desc = new TestDesc();

  test_desc->init();

  test_desc->weights_["weight"] = new device::Tensor("weight");
  test_desc->weights_["bias"] = new device::Tensor("bias");
  for (auto x : test_desc->weights_) {
    std::cout << x.first << std::endl;
  }
  test_desc->weights_["weight"]->reshape({32,1,3,3});
  // test_desc->weights_["weight"]->setName("weight");
  test_desc->weights_["bias"]->reshape({32});
  // test_desc->weights_["bias"]->setName("bias");
  test_net->setModelDesc(test_desc);
  test_net->init();
  test_net->dump(std::cout);
}
