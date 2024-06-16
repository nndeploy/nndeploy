
#include "nndeploy/net/test/test.h"

#include "nndeploy/device/tensor.h"

namespace nndeploy {

namespace net {

TestNet::TestNet() {
 

  auto input_expr = std::make_shared<op::Expr>("input");
  auto weight_expr = std::make_shared<op::Expr>("conv1.weight");

  auto model_desc = std::make_shared<op::ModelDesc>();


  

  auto conv1_p = std::make_shared<op::Conv2dParam>();
  conv1_p->padding = {1, 1};
  auto conv1 = model_desc->MakeConv2d(input_expr, weight_expr, conv1_p, "conv.1");
  auto relu1 = model_desc->MakeRelu(conv1,"relu.1");

  net_->setModelDesc(model_desc);

}

void TestNet::init(){
  net_->init();
  
}

}  // namespace net

}  // namespace nndeploy