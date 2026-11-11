#include <gtest/gtest.h>

#include "nndeploy/llm/sample.h"

namespace nndeploy {
namespace llm {
TEST(SamplerRegression, FiltersUseVocabularyElementsAndReturnOriginalIds) {
  auto *cpu = device::getDefaultHostDevice();
  float data[] = {-1000.f, -1000.f, 1000.f, -1000.f};
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNC, {1, 4});
  device::Tensor logits(cpu, desc, data, "logits");
  Sampler sampler("sample", {}, {});
  auto *param = static_cast<SampleParam *>(sampler.getParam());
  param->topK = 40;  // Larger than the entire vocabulary.
  for (const char *strategy : {"greedy", "temperature", "topK", "topP", "minP", "mixed"}) {
    param->sampler = strategy;
    param->mixed_samplers = {"topK", "topP", "temperature"};
    for (int i = 0; i < 20; ++i)
      EXPECT_EQ(sampler.sample(&logits), 2) << strategy;
  }
}

TEST(SamplerRegression, ProducesOneTokenPerBatchFromLastSequencePosition) {
  auto *cpu = device::getDefaultHostDevice();
  float data[] = {9, 0, 0, 0, 8, 0, 0, 7, 0, 0, 0, 6};
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatAuto, {2, 2, 3});
  device::Tensor logits(cpu, desc, data, "logits");
  dag::Edge input("logits"), output("tokens");
  Sampler sampler("sample", {&input}, {&output});
  static_cast<SampleParam *>(sampler.getParam())->sampler = "greedy";
  auto *param = static_cast<SampleParam *>(sampler.getParam());
  param->temperature = 0;
  EXPECT_EQ(sampler.sample(&logits), 0);
  ASSERT_TRUE(input.set(&logits, true));
  ASSERT_TRUE(sampler.run());
  auto *tokens = output.getGraphOutput<tokenizer::TokenizerIds>();
  ASSERT_NE(tokens, nullptr);
  EXPECT_EQ(tokens->ids_, (std::vector<std::vector<int>>{{1}, {2}}));
}
}  // namespace llm
}  // namespace nndeploy
