#include <gtest/gtest.h>

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference.h"

namespace nndeploy {
namespace inference {

TEST(InferenceParamSerialization, EmptyPathPlaceholdersClearPreviousPaths) {
  InferenceParam param;
  param.model_value_ = {"previous.onnx"};
  param.external_model_data_ = {"previous.data"};
  rapidjson::Document json;
  json.Parse(R"({"is_path_":true,"model_value_":[""],"external_model_data_":[""]})");
  ASSERT_FALSE(json.HasParseError());
  ASSERT_EQ(param.deserialize(json), base::kStatusCodeOk);
  EXPECT_TRUE(param.model_value_.empty());
  EXPECT_TRUE(param.external_model_data_.empty());

  json.Parse(R"({"is_path_":true,"model_value_":["model.onnx"],"external_model_data_":["weights.data"]})");
  ASSERT_EQ(param.deserialize(json), base::kStatusCodeOk);
  EXPECT_EQ(param.model_value_, std::vector<std::string>({"model.onnx"}));
  EXPECT_EQ(param.external_model_data_,
            std::vector<std::string>({"weights.data"}));
}

TEST(DefaultInferenceLifecycle, CallerRetainsExternalStreams) {
  auto *cpu = device::getDefaultHostDevice();
  std::unique_ptr<device::Stream> first(cpu->createStream());
  std::unique_ptr<device::Stream> second(cpu->createStream());
  {
    auto engine = createInference(base::kInferenceTypeDefault);
    ASSERT_NE(engine, nullptr);
    engine->setStream(first.get());
    engine->setStream(first.get());
    engine->setStream(second.get());
    EXPECT_EQ(first->synchronize(), base::kStatusCodeOk);
    ASSERT_NE(engine->init(), base::kStatusCodeOk);
    EXPECT_EQ(engine->deinit(), base::kStatusCodeOk);
    EXPECT_EQ(engine->deinit(), base::kStatusCodeOk);
  }
  EXPECT_EQ(second->synchronize(), base::kStatusCodeOk);
}

}  // namespace inference
}  // namespace nndeploy
