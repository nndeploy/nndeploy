#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "nndeploy/qwen/qwen.h"

namespace nndeploy {
TEST(QwenSampler, RejectsInvalidHistoryAndSelectsLargestNegativeScore) {
  float values[] = {100, 0, 0, -5, -1, -1.05f};
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatAuto,
                          {2, 3});
  device::Tensor tensor(device::getDefaultHostDevice(), desc, values, "scores");
  dag::Edge logits("logits"), history("history"), output("output");
  qwen::PrefillSampleNode prefill("prefill", {&logits, &history}, {&output});
  tokenizer::TokenizerIds ids;
  ASSERT_TRUE(logits.set(&tensor, true));
  ASSERT_TRUE(history.set(&ids, true));
  EXPECT_FALSE(prefill.run());
  for (int bad : {-1, 3}) {
    ids.ids_ = {{bad}};
    EXPECT_FALSE(prefill.run());
  }
  ids.ids_ = {{0}};
  ASSERT_TRUE(prefill.run());
  auto* result = dynamic_cast<tokenizer::TokenizerIds*>(output.getGraphOutputParam());
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->ids_.at(0), (std::vector<int>{1}));

  dag::Edge decoded("decoded");
  qwen::DecodeSampleNode decode("decode", {&logits}, {&decoded});
  auto* param = static_cast<qwen::DecodeSampleParam*>(decode.getParam());
  param->history_ids_.ids_ = {{}};
  EXPECT_FALSE(decode.run());
  param->history_ids_.ids_ = {{1}};
  ASSERT_TRUE(decode.run());
  EXPECT_EQ(param->history_ids_.ids_[0], (std::vector<int>{1, 2}));
  EXPECT_FLOAT_EQ(values[4], -1.0f);
}

TEST(QwenTokenizer, PrefillAndDecodeInitializeFromConfiguredFile) {
  const auto &backends = inference::getGlobalInferenceCreatorMap();
  if (backends.empty()) GTEST_SKIP() << "Qwen requires an inference backend";
  namespace fs = std::filesystem;
  const auto dir = fs::temp_directory_path() / ("qwen-" + base::getUniqueString());
  ASSERT_TRUE(fs::create_directory(dir));
  struct Cleanup {
    fs::path dir;
    ~Cleanup() { std::error_code ec; fs::remove_all(dir, ec); }
  } cleanup{dir};
  const auto tokenizer_path = (dir / "tokenizer.json").generic_string();
  const auto stops_path = (dir / "stops.txt").generic_string();
  const auto config_path = (dir / "qwen.json").string();
  std::ofstream(tokenizer_path) << R"({"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}})";
  std::ofstream(stops_path) << "tokens\n0 0 0\n\n";
  std::ofstream(config_path) << R"({"layer_nums":1,"hidden_size":1,"max_seq_len":2,)"
      << R"("model_path":"model","embedding_file":"embedding","prompt_template":"",)"
      << R"("prompt":"","key_value_shape":[],"tokenizer_json":")" << tokenizer_path
      << R"(","tokenizer_txt":")" << stops_path << R"("})";

  auto check = [&](auto &stage, const char *token_node) {
    stage.setConfigPath(config_path);
    ASSERT_TRUE(stage.setInferenceType(backends.begin()->first));
    ASSERT_TRUE(stage.setConfigParam());
    auto *node = stage.getNode(token_node);
    ASSERT_NE(node, nullptr);
    ASSERT_TRUE(node->init());
    EXPECT_TRUE(node->deinit());
  };
  qwen::QwenPrefill prefill("prefill", {}, {});
  check(prefill, "token_node");
  qwen::QwenDecode decode("decode", {}, {});
  check(decode, "decode_node");
}
}  // namespace nndeploy
