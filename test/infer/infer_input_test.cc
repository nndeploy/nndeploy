#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>

#include "nndeploy/infer/infer.h"

namespace nndeploy {
namespace {

class ProbeInference : public inference::Inference {
 public:
  using Inference::Inference;
  base::Status init() override { return base::kStatusCodeOk; }
  base::Status deinit() override { return base::kStatusCodeOk; }
  bool isInputDynamic() override { return true; }
  base::Status reshape(base::ShapeMap &) override { return shape_status; }
  base::Status setInputTensor(const std::string &, device::Tensor *) override {
    ++set_calls;
    return input_status;
  }
  base::Status run() override {
    ++run_calls;
    return base::kStatusCodeOk;
  }
  device::Tensor *getOutputTensorAfterRun(const std::string &, base::DeviceType,
                                         bool, base::DataFormat) override {
    return nullptr;
  }
  base::Status shape_status, input_status;
  int set_calls = 0, run_calls = 0;
};

TEST(InferInput, RejectsMissingTensorAndPropagatesBackendErrors) {
  auto type = base::kInferenceTypeDefault;
  auto &creator = inference::getGlobalInferenceCreatorMap().at(type);
  auto original = creator;
  creator = std::make_shared<inference::TypeInferenceCreator<ProbeInference>>();
  dag::Edge input("x");
  infer::Infer node("probe", {&input}, {}, type);
  creator = original;
  ASSERT_EQ(node.init(), base::kStatusCodeOk);
  auto probe = std::dynamic_pointer_cast<ProbeInference>(node.getInference());
  ASSERT_TRUE(probe);
  EXPECT_EQ(node.run(), base::kStatusCodeErrorNullParam);
  device::Tensor tensor("x");
  ASSERT_EQ(input.set(&tensor, true), base::kStatusCodeOk);
  probe->shape_status = base::kStatusCodeErrorInvalidValue;
  EXPECT_EQ(node.run(), base::kStatusCodeErrorInvalidValue);
  EXPECT_EQ(probe->set_calls, 0);
  probe->shape_status = base::kStatusCodeOk;
  probe->input_status = base::kStatusCodeErrorInvalidParam;
  EXPECT_EQ(node.run(), base::kStatusCodeErrorInvalidParam);
  EXPECT_EQ(probe->run_calls, 0);
  probe->input_status = base::kStatusCodeOk;
  EXPECT_EQ(node.run(), base::kStatusCodeOk);
  EXPECT_EQ(probe->run_calls, 1);
  EXPECT_EQ(node.deinit(), base::kStatusCodeOk);
}


class CountedOutput : public device::Tensor {
 public:
  explicit CountedOutput(std::shared_ptr<std::atomic<int>> live) : live_(std::move(live)) { ++*live_; }
  ~CountedOutput() override { --*live_; }
 private:
  std::shared_ptr<std::atomic<int>> live_;
};

class OutputProbeInference : public ProbeInference {
 public:
  explicit OutputProbeInference(base::InferenceType type) : ProbeInference(type) {
    output_tensors_["y"] = &internal;
    output_tensors_["z"] = &internal;
  }
  bool isInputDynamic() override { return false; }
  device::Tensor *getOutputTensorAfterRun(const std::string &, base::DeviceType,
                                         bool, base::DataFormat) override {
    ++output_calls;
    return borrowed ? &internal : new CountedOutput(live);
  }
  device::Tensor internal;
  std::shared_ptr<std::atomic<int>> live = std::make_shared<std::atomic<int>>(0);
  bool borrowed = false;
  int output_calls = 0;
};

TEST(InferInput, CancelledOutputReleasesOwnedTensorAndPropagatesFailure) {
  using namespace std::chrono_literals;
  for (bool borrowed : {false, true}) {
    auto type = base::kInferenceTypeDefault;
    auto &creator = inference::getGlobalInferenceCreatorMap().at(type);
    auto original = creator;
    creator = std::make_shared<inference::TypeInferenceCreator<OutputProbeInference>>();
    dag::Edge output("y"), later("z");
    output.setParallelType(base::kParallelTypePipeline);
    output.setQueueMaxSize(1);
    output.setQueueOverflowPolicy(base::kQueueOverflowPolicyAllBackpressure);
    output.markGraphOutput();
    int queued = 1;
    ASSERT_TRUE(output.set(&queued, true));
    infer::Infer node("cancel_output", {}, {&output, &later}, type);
    creator = original;
    ASSERT_TRUE(node.init());
    auto probe = std::dynamic_pointer_cast<OutputProbeInference>(node.getInference());
    probe->borrowed = borrowed;
    auto run = std::async(std::launch::async, [&] { return node.run(); });
    EXPECT_EQ(run.wait_for(50ms), std::future_status::timeout);
    output.requestTerminate();
    ASSERT_EQ(run.wait_for(1s), std::future_status::ready);
    EXPECT_EQ(run.get(), base::kStatusCodeErrorInvalidValue);
    EXPECT_EQ(probe->output_calls, 1);
    EXPECT_EQ(probe->live->load(), 0);
    EXPECT_TRUE(later.empty());
    EXPECT_TRUE(node.deinit());
  }
}

}  // namespace
}  // namespace nndeploy
