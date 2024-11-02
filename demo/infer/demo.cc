#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"
#include "nndeploy/ir/op_param.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/op.h"
#include "safetensors.hh"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/schema.h"
using namespace nndeploy;

// std::shared_ptr<Expr> makeBlock(ir::ModelDesc *model_desc,
//                                 std::shared_ptr<ir::ModelDesc> model_block) {
//   if (model_desc != nullptr) {
//     model_desc->blocks_.push_back(model_block);
//   }
//   auto expr = std::make_shared<Expr>(model_block);
//   return expr;
// }

class TestDesc1 : public ir::ModelDesc {
 public:
  void init() {
    auto input =
        op::makeInput(this, "input", base::dataTypeOf<float>(), {1, 1, 8, 8});
    auto conv_param = std::make_shared<ir::ConvParam>();
    conv_param->kernel_shape_ = {3, 3};
    auto conv1 = op::makeConv(this, input, conv_param, "weight", "bias");
    auto relu1 = op::makeRelu(this, conv1);
    auto relu2 = op::makeRelu(this, relu1);
    // auto softmax = op::makeSoftMax(this, relu1,
    // std::make_shared<ir::SoftmaxParam>()); op::makeOutput(this, softmax);
    op::makeOutput(this, relu1);
    op::makeOutput(this, relu2);
  }
};
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

    // auto softmax = op::makeSoftMax(this, relu1,
    // std::make_shared<ir::SoftmaxParam>()); op::makeOutput(this, softmax);
    // auto test_block = new TestDesc1();

    op::makeOutput(this, relu2);
  }
};

void printHelloWorld() { std::cout << "hello world!" << std::endl; }

void test_onnx_schemas() {
  std::cout << (onnx::OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() ==
                0)
            << std::endl;

  // onnx::RegisterOnnxOperatorSetSchema();
  auto schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema("Conv");
  std::cout << schema << std::endl;
  std::cout << schema->Name() << std::endl;
  for (const auto it : schema->attributes()) {
    std::cout << it.first << " " /* << it.second.description << " " */
              << static_cast<int>(it.second.type) << std::endl;
  }
}

void testOnnxImport() {
  std::shared_ptr<ir::OnnxInterpret> onnx_int =
      std::make_shared<ir::OnnxInterpret>();
  onnx_int->interpret(
      {"/Users/realtyxxx_mac/study/gt/conv_relu.onnx"},
      // onnx_int->interpret({"/Users/realtyxxx_mac/study/gt/resnet18-v1-7.onnx"},
      {nndeploy::ir::ValueDesc("input")});
}

int main(int argc, char const *argv[]) {
  // printHelloWorld();
  // test_onnx_schemas();
  // testOnnxImport();
  std::string sep = "-";
  for (int i = 0; i < 5; ++i) sep += sep;
  sep = "\n" + sep + "\n";

  auto test_desc = new TestDesc();
  test_desc->init();

  base::DeviceType device_type = base::kDeviceTypeCodeCpu;
  device_type.device_id_ = 0;
  auto device = device::getDevice(device_type);

  device::TensorDesc weight_desc(base::dataTypeOf<float>(),
                                 base::kDataFormatOIHW, {32, 1, 3, 3});
  test_desc->weights_["weight"] =
      new device::Tensor(device, weight_desc, "weight");
  device::TensorDesc bias_desc(base::dataTypeOf<float>(), base::kDataFormatN,
                               {32});
  test_desc->weights_["bias"] = new device::Tensor(device, bias_desc, "bias");

  std::cout << sep << "desc dump : " << std::endl;
  test_desc->dump(std::cout);

#if 0
  {
    // {
    // auto test_net = std::make_shared<net::Net>();
    // test_net->setModelDesc(test_desc);
    // test_net->setDeviceType(device_type);
    // test_net->init();

    // std::cout << sep << "net dump : " << std::endl;
    // test_net->dump(std::cout);
    // }
    std::ofstream test_model_txt(
        "/Users/realtyxxx_mac/study/gt/test.model.txt");
    // test_net->dump(test_model_txt);
    // test_net->saveModelUseSafetensors(std::cout,
    // "/Users/realtyxxx_mac/study/gt/test.safetensors");
    // safetensors::safetensors_t st;
    // test_desc->serializeWeightsToSafetensors(st);
    // std::string warn, err;
    // safetensors::save_to_file(
    //     st, "/Users/realtyxxx_mac/study/gt/test.safetensors", &warn, &err);

    std::shared_ptr<ir::DefaultInterpret> default_interpret_0 =
        std::make_shared<ir::DefaultInterpret>();
    default_interpret_0->model_desc_ = test_desc;

    default_interpret_0->saveModelUseSafetensors(
        test_model_txt, "/Users/realtyxxx_mac/study/gt/test.safetensors");

    std::shared_ptr<ir::DefaultInterpret> default_interpret_1 =
        std::make_shared<ir::DefaultInterpret>();
    default_interpret_1->interpret(
        {std::string("/Users/realtyxxx_mac/study/gt/test.model.txt"),
         std::string("/Users/realtyxxx_mac/study/gt/test.safetensors")});

    std::cout << sep << "from load" << std::endl;
    default_interpret_1->model_desc_->dump(std::cout);
    std::cout << sep << "weight" << std::endl;
    auto& weights = default_interpret_1->model_desc_->weights_;
    for (auto weight : weights) {
      std::cout << weight.first << " " << weight.second << std::endl;
    }
  }
#endif
  {
    std::string weight_path =
        "/Users/realtyxxx_mac/study/gt/"
        "rtdetr_save_out.safetensors";
    // "/Users/realtyxxx_mac/.cache/huggingface/hub/"
    // "models--PekingU--rtdetr_r18vd/snapshots/"
    // "ac77a11ff0170a41b771c03264987f8ce2b0d753/model.safetensors";
    auto new_desc = new ir::ModelDesc();

    std::ifstream inFile("/Users/realtyxxx_mac/study/gt/rtdetr.txt");

    // 检查文件是否成功打开
    if (!inFile) {
      std::cerr << "无法打开文件！" << std::endl;
      return 1;
    }

    std::string line;
    // 使用 std::getline 逐行读取文件内容
    while (std::getline(inFile, line)) {
      std::cout << line << std::endl;  // 输出每行内容到控制台
      new_desc->weights_.insert({line, new device::Tensor(line)});
    }
    std::cout << sep << "new_desc deserialize : " << std::endl;

    new_desc->deserializeWeightsFromSafetensors(weight_path);

    std::cout << sep << "new_desc serialize : " << std::endl;
    std::shared_ptr<safetensors::safetensors_t> st_ptr;
    new_desc->serializeWeightsToSafetensors(st_ptr);
    std::string err, warn;
    safetensors::save_to_file(
        *st_ptr, "/Users/realtyxxx_mac/study/gt/rtdetr_out.safetensors", &warn,
        &err);

    // 关闭文件
    inFile.close();
  }

  return 0;
}
