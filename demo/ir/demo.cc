#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/op.h"

using namespace nndeploy;

class ExprDemo : public ir::ModelDesc {
 public:
  ExprDemo(){};
  ~ExprDemo(){};
  void init() {
    auto input = op::makeInput(this, "input", base::dataTypeOf<float>(),
                               {1, 3, 640, 640});
    auto pool_param = std::make_shared<ir::MaxPoolParam>();
    pool_param->kernel_shape_ = {2, 2};
    pool_param->strides_ = {2, 2};
    auto pool1 = op::makeMaxPool(this, input, pool_param);
    auto relu1 = op::makeRelu(this, pool1);
    auto softmax_0 =
        op::makeSoftmax(this, relu1, std::make_shared<ir::SoftmaxParam>());
    auto softmax_1 =
        op::makeSoftmax(this, relu1, std::make_shared<ir::SoftmaxParam>());

    auto add = op::makeAdd(this, softmax_0, softmax_1);

    op::makeOutput(this, add);
  }
};

int main(int argc, char const *argv[]) {
  ExprDemo expr_demo;
  expr_demo.init();
  expr_demo.serializeStructureToJson("expr_demo.ir.json");

  auto net = std::make_shared<net::Net>();
  net->setModelDesc(&expr_demo);

  base::DeviceType device_type;
  // device_type.code_ = base::kDeviceTypeCodeCpu;
  device_type.code_ = base::kDeviceTypeCodeAscendCL;
  device_type.device_id_ = 0;
  net->setDeviceType(device_type);

  net->init();

  net->dump(std::cout);

  return 0;
}
