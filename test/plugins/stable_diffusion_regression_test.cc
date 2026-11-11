#include <gtest/gtest.h>
#include "nndeploy/stable_diffusion/text2image.h"
#include "nndeploy/stable_diffusion/clip.h"
#include "nndeploy/stable_diffusion/denoise.h"
#include "nndeploy/stable_diffusion/vae.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/infer/infer.h"

#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
TEST(DDIMScheduler, RejectsInvalidStepsAndProducesRequestedCount) {
  using namespace stable_diffusion;
  DDIMScheduler scheduler(kSchedulerTypeDDIM);
  DDIMSchedulerParam param;
  param.num_train_timesteps_ = 10;
  for (int steps : {-1, 0, 11}) {
    param.num_inference_steps_ = steps;
    EXPECT_FALSE(scheduler.init(&param));
  }
  param.num_inference_steps_ = 3;
  ASSERT_TRUE(scheduler.init(&param));
  ASSERT_TRUE(scheduler.setTimesteps());
  EXPECT_EQ(scheduler.getTimesteps(), (std::vector<int>{6, 3, 0}));
  ASSERT_TRUE(scheduler.deinit());
  EXPECT_FALSE(scheduler.setTimesteps());
}

TEST(ClipTokens, TruncatesLongPromptAndPadsEmptyPrompt) {
  dag::Edge input("ids"), output("tokens");
  std::unique_ptr<dag::Node> node(dag::createNode(
      "nndeploy::stable_diffusion::CvtTokenIds2Tensor", "clip", {&input}, {&output}));
  ASSERT_NE(node, nullptr);
  tokenizer::TokenizerIds ids;
  ASSERT_TRUE(input.set(&ids, true));
  EXPECT_FALSE(node->run());
  for (int length : {0, 75, 76, 200}) {
    ids.ids_ = {std::vector<int32_t>(length, 42)};
    ASSERT_TRUE(node->run());
    auto *tensor = output.getGraphOutputTensor();
    ASSERT_NE(tensor, nullptr);
    const auto *values = static_cast<const int32_t *>(tensor->getData());
    ASSERT_EQ(tensor->getSize(), 77 * sizeof(int32_t));
    EXPECT_EQ(values[0], 49406);
    for (int i = 1; i < 77; ++i)
      EXPECT_EQ(values[i], i <= std::min(length, 75) ? 42 : 49407);
  }
}

TEST(DDIMScheduler, RejectsShortBuffersBeforeWriting) {
  using namespace stable_diffusion;
  DDIMScheduler scheduler(kSchedulerTypeDDIM);
  DDIMSchedulerParam param;
  ASSERT_TRUE(scheduler.init(&param));
  ASSERT_TRUE(scheduler.setTimesteps());
  auto *host = device::getDefaultHostDevice();
  device::TensorDesc one(base::dataTypeOf<float>(), base::kDataFormatAuto, {1});
  device::TensorDesc two(base::dataTypeOf<float>(), base::kDataFormatAuto, {2});
  float values[] = {1, 2}, zero = 0, sentinel = 9;
  device::Tensor sample(host, two, values, "sample"), timestep(host, one, &zero, "step"),
      output(host, one, &sentinel, "output"), valid(host, two), empty("empty");
  EXPECT_FALSE(scheduler.step(&sample, &timestep, &sample, &output));
  EXPECT_FLOAT_EQ(sentinel, 9);
  EXPECT_FALSE(scheduler.step(&sample, &empty, &sample, &valid));
  ASSERT_TRUE(scheduler.step(&sample, &timestep, &sample, &valid));
  ASSERT_TRUE(scheduler.deinit());
}

TEST(ScaleLatents, RepeatedTeardownAndReinitialization) {
  std::unique_ptr<dag::Node> node(dag::createNode(
      "nndeploy::stable_diffusion::ScaleLatents", "scale", {}, {}));
  ASSERT_NE(node, nullptr);
  EXPECT_TRUE(node->deinit());
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(node->init());
    EXPECT_TRUE(node->deinit());
    EXPECT_TRUE(node->deinit());
  }
}

TEST(DDIMSchedule, StandaloneNodeRejectsRunBeforeInitAndInitializesScheduler) {
  std::unique_ptr<dag::Node> node(dag::createNode(
      "nndeploy::stable_diffusion::DDIMSchedule", "schedule", {}, {}));
  ASSERT_NE(node, nullptr);
  EXPECT_FALSE(node->run());
  EXPECT_TRUE(node->init());
  EXPECT_TRUE(node->deinit());
}

TEST(DDIMSchedule, UnguidedStepLeavesUpstreamTensorAliveAfterTeardown) {
  dag::Edge prediction("prediction"), previous("previous"), step("step"), output("output");
  std::unique_ptr<dag::Node> node(dag::createNode(
      "nndeploy::stable_diffusion::DDIMSchedule", "schedule",
      {&prediction, &previous, &step}, {&output}));
  auto *param = dynamic_cast<stable_diffusion::DDIMSchedulerParam*>(node->getParam());
  ASSERT_NE(param, nullptr);
  param->guidance_scale_ = 1;
  param->image_width_ = param->image_height_ = 8;
  param->unet_channels_ = 1;
  ASSERT_TRUE(node->init());
  auto *host = device::getDefaultHostDevice();
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCHW, {1, 1, 1, 1});
  float value = 3, zero = 0;
  device::Tensor pred(host, desc, &value, "pred"), t(host, desc, &zero, "step");
  ASSERT_TRUE(prediction.set(&pred, true));
  ASSERT_TRUE(previous.set(&pred, true));
  ASSERT_TRUE(step.set(&t, true));
  ASSERT_TRUE(node->run());
  ASSERT_NE(output.getGraphOutputTensor(), nullptr);
  ASSERT_TRUE(node->deinit());
  EXPECT_EQ(pred.getData(), &value);
}

namespace {
class FailingUnet : public inference::Inference {
 public:
  using Inference::Inference;
  base::Status init() override { return base::kStatusCodeOk; }
  base::Status deinit() override { return base::kStatusCodeOk; }
  base::Status reshape(base::ShapeMap&) override { return base::kStatusCodeOk; }
  std::vector<std::string> getAllInputTensorName() override {
    return {"embeddings", "cfg_latents", "timestep"};
  }
  base::Status setInputTensor(const std::string&, device::Tensor*) override {
    return base::kStatusCodeOk;
  }
  base::Status run() override { ++calls; return base::kStatusCodeErrorIO; }
  device::Tensor* getOutputTensorAfterRun(const std::string&, base::DeviceType,
                                         bool, base::DataFormat) override { return nullptr; }
  int calls = 0;
};
}

TEST(Denoise, ReturnsUnetFailureOnFirstIteration) {
  dag::Edge inputs("initial"), embeddings("conditioning"), output("out");
  std::unique_ptr<dag::Node> node(dag::createNode("nndeploy::stable_diffusion::Denoise",
      "denoise", {&inputs, &embeddings}, {&output}));
  ASSERT_NE(node, nullptr);
  ASSERT_TRUE(node->defaultParam());
  auto* composite = dynamic_cast<dag::CompositeNode*>(node.get());
  auto* infer = dynamic_cast<infer::Infer*>(composite->getNode("unet_infer"));
  ASSERT_NE(infer, nullptr);
  auto& creators = inference::getGlobalInferenceCreatorMap();
  auto type = base::kInferenceTypeDefault;
  auto original = creators.find(type);
  auto saved = original == creators.end() ? nullptr : original->second;
  creators[type] = std::make_shared<inference::TypeInferenceCreator<FailingUnet>>();
  auto status = infer->setInferenceType(type);
  if (saved) creators[type] = saved;
  else creators.erase(type);
  ASSERT_TRUE(status);
  auto* param = dynamic_cast<stable_diffusion::DDIMSchedulerParam*>(
      composite->getNode("ddim_schedule")->getParam());
  param->image_width_ = param->image_height_ = 8;
  param->unet_channels_ = 1;
  param->guidance_scale_ = 1;
  param->num_inference_steps_ = 2;
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCHW, {1, 1, 1, 1});
  device::Tensor data(device::getDefaultHostDevice(), desc);
  ASSERT_TRUE(inputs.set(&data, true));
  ASSERT_TRUE(embeddings.set(&data, true));
  ASSERT_TRUE(node->init());
  EXPECT_EQ(node->run(), base::kStatusCodeErrorIO);
  EXPECT_EQ(std::dynamic_pointer_cast<FailingUnet>(infer->getInference())->calls, 1);
  EXPECT_TRUE(node->deinit());
}

TEST(TensorToMat, RejectsUnsupportedChannelsAndBatchCounts) {
  for (auto shape : {base::IntVector{1, 4, 1, 1}, {2, 3, 1, 1}, {0, 3, 1, 1}}) {
    dag::Edge input("tensor"), output("image");
    std::unique_ptr<dag::Node> node(dag::createNode(
        "nndeploy::stable_diffusion::TensorToMat", "convert", {&input}, {&output}));
    ASSERT_NE(node, nullptr);
    float values[6] = {};
    device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCHW, shape);
    device::Tensor tensor(device::getDefaultHostDevice(), desc, values, "tensor");
    ASSERT_TRUE(input.set(&tensor, true));
    EXPECT_FALSE(node->run());
  }
}

TEST(TensorToMat, PreservesPixelOrderAndConvertsRgbToBgr) {
  for (bool planar : {true, false}) {
    dag::Edge input("tensor"), output("image");
    std::unique_ptr<dag::Node> node(dag::createNode(
        "nndeploy::stable_diffusion::TensorToMat", "convert", {&input}, {&output}));
    ASSERT_NE(node, nullptr);
    float nchw[] = {1, -1, -1, 0, 0, 1}, nhwc[] = {1, -1, 0, -1, 0, 1};
    device::TensorDesc desc(base::dataTypeOf<float>(),
        planar ? base::kDataFormatNCHW : base::kDataFormatNHWC,
        planar ? base::IntVector{1, 3, 1, 2} : base::IntVector{1, 1, 2, 3});
    device::Tensor tensor(device::getDefaultHostDevice(), desc, planar ? nchw : nhwc, "rgb");
    ASSERT_TRUE(input.set(&tensor, true));
    ASSERT_TRUE(node->run());
    auto* image = output.getGraphOutputCvMat();
    ASSERT_NE(image, nullptr);
    EXPECT_EQ(image->at<cv::Vec3b>(0, 0), cv::Vec3b(128, 0, 255));
    EXPECT_EQ(image->at<cv::Vec3b>(0, 1), cv::Vec3b(255, 128, 0));
  }
}

TEST(TextToImageGraph, RejectsMissingModelsBeforeBuildingNodes) {
  using namespace stable_diffusion;
  Text2ImageParam models;
  DDIMSchedulerParam scheduler;
  std::vector<base::Param*> params{&models, &scheduler};
  const auto backend = base::kInferenceTypeDefault;
  dag::Edge input("in"), negative("negative"), output("out");
  for (size_t n = 0; n < 4; ++n) {
    models.model_value_.assign(n, "model");
    EXPECT_EQ(createStableDiffusionText2ImageGraph("sd", "", "", backend,
              kSchedulerTypeDDIM, params, 1), nullptr);
    if (n < 2) EXPECT_EQ(createCLIPGraph("clip", &input, &negative, &output, backend, params), nullptr);
    if (n < 3) EXPECT_EQ(createDenoiseGraph("denoise", &input, &output,
                          kSchedulerTypeDDIM, backend, params, 1), nullptr);
    EXPECT_EQ(createVAEGraph("vae", &input, &output, backend, params), nullptr);
  }
  models.model_value_ = {"tokenizer", "clip", "", "vae"};
  EXPECT_EQ(createStableDiffusionText2ImageGraph("sd", "", "", backend,
            kSchedulerTypeDDIM, params, 1), nullptr);
  models.model_value_[2] = "unet";
  std::unique_ptr<dag::Graph> graph(createStableDiffusionText2ImageGraph(
      "sd", "", "", backend, kSchedulerTypeDDIM, params, 1));
  EXPECT_NE(graph, nullptr);
}
}  // namespace nndeploy
