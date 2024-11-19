
// #include "test.h"

// #include "nndeploy/device/tensor.h"

// namespace nndeploy {

// namespace net {

// TestNet::TestNet() {
//   // auto input_expr = std::make_shared<op::Expr>("input");
//   // auto weight_expr = std::make_shared<op::Expr>("conv1.weight");

//   // auto model_desc = std::make_shared<ir::ModelDesc>();

//   // auto conv1_p = std::make_shared<ir::ConvParam>();
//   // conv1_p->padding = {1, 1};
//   // auto conv1 =
//   //     model_desc->MakeConv(input_expr, weight_expr, conv1_p, "conv.1");
//   // // TODO： 这样是不是更加简洁？
//   // // auto conv1 =
//   // //     model_desc->MakeConv("input_expr", "weight_expr", conv1_p,
//   // "conv.1"); auto relu1 = model_desc->MakeRelu(conv1, "relu.1");

//   // net_->setModelDesc(model_desc);
// }

// void TestNet::init() { net_->init(); }

// }  // namespace net

// }  // namespace nndeploy