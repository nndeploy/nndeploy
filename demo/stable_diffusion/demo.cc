#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/framework.h"
#include "nndeploy/stable_diffusion/text2image.h"
#include "nndeploy/tokenizer/tokenizer.h"

using namespace nndeploy;

int main(int argc, char const* argv[]) {
  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  const std::string name = "txt2img_sd";
  base::InferenceType clip_inference_type =
      nndeploy::base::kInferenceTypeOnnxRuntime;
  base::InferenceType unet_inference_type =
      nndeploy::base::kInferenceTypeOnnxRuntime;
  base::InferenceType vae_inference_type =
      nndeploy::base::kInferenceTypeOnnxRuntime;
  stable_diffusion::SchedulerType scheduler_type =
      stable_diffusion::kSchedulerTypeDDIM;
  std::vector<base::Param*> param;

  dag::Edge* prompt = new dag::Edge("prompt");
  dag::Edge* negative_prompt = new dag::Edge("negative_prompt");

  dag::Graph* graph = stable_diffusion::createStableDiffusionText2ImageGraph(
      name, prompt, negative_prompt, clip_inference_type, unet_inference_type,
      vae_inference_type, scheduler_type, param);

  base::Status status = graph->setParallelType(base::kParallelTypeSequential);

  std::cout << "1" << std::endl;
  graph->setTimeProfileFlag(true);

  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  tokenizer::TokenizerText* prompt_text = new tokenizer::TokenizerText();
  prompt_text->texts_ = {"a cute cat"};
  prompt->set(prompt_text, 0, true);

  tokenizer::TokenizerText* negative_prompt_text =
      new tokenizer::TokenizerText();
  negative_prompt_text->texts_ = {""};
  negative_prompt->set(negative_prompt_text, 0, true);

  std::cout << "2" << std::endl;

  NNDEPLOY_TIME_POINT_START("graph->dump()");
  status = graph->dump();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph dump failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->dump()");

  NNDEPLOY_TIME_POINT_START("graph->run()");
  status = graph->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph run failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->run()");

  NNDEPLOY_TIME_POINT_START("graph->deinit()");
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->deinit()");

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}