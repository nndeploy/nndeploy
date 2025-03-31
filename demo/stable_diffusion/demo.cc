#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/framework.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/text2image.h"
#include "nndeploy/tokenizer/tokenizer.h"

using namespace nndeploy;
using namespace nndeploy::stable_diffusion;

DEFINE_string(prompt, "a small cute dog", "prompt for txt2img");

std::string getPrompt() { return FLAGS_prompt; }

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  // name of Stable Diffusion DAG: txt2img
  std::string name = demo::getName();
  // 推理后端类型，例如:
  // kInferenceTypeOpenVino / kInferenceTypeTensorRt / kInferenceTypeOnnxRuntime
  // base::InferenceType inference_type = demo::getInferenceType();
  base::InferenceType inference_type = base::kInferenceTypeOnnxRuntime;
  // 推理设备类型，例如:
  // kDeviceTypeCodeX86:0/kDeviceTypeCodeCuda:0/...
  base::DeviceType device_type = demo::getDeviceType();
  // 模型类型，例如:
  // kModelTypeOnnx/kModelTypeMnn/...
  // base::ModelType model_type = demo::getModelType();
  base::ModelType model_type = base::kModelTypeOnnx;
  // 模型是否是路径
  bool is_path = demo::isPath();
  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();
  // output path
  std::string output_path = demo::getOutputPath();
  // base::kParallelTypePipeline / base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();
  std::string text = getPrompt();
  NNDEPLOY_LOGI("prompt = %s.\n", text.c_str());

  stable_diffusion::SchedulerType scheduler_type =
      stable_diffusion::kSchedulerTypeDDIM;

  std::vector<base::Param*> param;
  Text2ImageParam* text2image_param = new Text2ImageParam();
  text2image_param->inference_type_ = inference_type;
  text2image_param->device_type_ = device_type;
  text2image_param->model_type_ = model_type;
  text2image_param->is_path_ = is_path;
  text2image_param->model_value_ = model_value;
  text2image_param->output_path_ = output_path;
  text2image_param->pt_ = pt;
  param.push_back(text2image_param);

  DDIMSchedulerParam* ddim_param = new DDIMSchedulerParam();
  ddim_param->version_ = "v1.5";
  ddim_param->unet_channels_ = 4;
  ddim_param->image_height_ = 512;
  ddim_param->image_width_ = 512;
  ddim_param->num_train_timesteps_ = 1000;
  ddim_param->clip_sample_ = false;
  ddim_param->num_inference_steps_ = 50;
  ddim_param->guidance_scale_ = 7.5;
  ddim_param->vae_scale_factor_ = 0.18215;
  ddim_param->init_noise_sigma_ = 1.0f;
  ddim_param->beta_start_ = 0.00085;
  ddim_param->beta_end_ = 0.012;
  ddim_param->beta_schedule_ = "scaled_linear";
  ddim_param->eta_ = 0.0;
  ddim_param->set_alpha_to_one_ = false;
  param.push_back(ddim_param);

  dag::Edge* prompt = new dag::Edge("prompt");
  dag::Edge* negative_prompt = new dag::Edge("negative_prompt");

  dag::Graph* graph = stable_diffusion::createStableDiffusionText2ImageGraph(
      name, prompt, negative_prompt, inference_type, inference_type,
      inference_type, scheduler_type, param);

  base::Status status = graph->setParallelType(pt);

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
  prompt_text->texts_ = {text};
  prompt->set(prompt_text, 0, true);

  tokenizer::TokenizerText* negative_prompt_text =
      new tokenizer::TokenizerText();
  negative_prompt_text->texts_ = {""};
  negative_prompt->set(negative_prompt_text, 0, true);

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

  NNDEPLOY_TIME_PROFILER_PRINT("demo");

  delete text2image_param;
  delete prompt_text;
  delete negative_prompt_text;
  delete ddim_param;
  delete prompt;
  delete negative_prompt;
  delete graph;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}