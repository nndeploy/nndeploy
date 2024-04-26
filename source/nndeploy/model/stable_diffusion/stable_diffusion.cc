#include "nndeploy/model/stable_diffusion/stable_diffusion.h"

#include "nndeploy/base/opencv_include.h"
#include "nndeploy/model/infer.h"
#include "nndeploy/model/preprocess/cvtcolor_resize_pad.h"

// TODO: edge要支持能够同名的情况

namespace nndeploy {
namespace model {

/**
 * @brief
 *
 * @param name
 * @param input
 * @param output
 * @param param
 * @return dag::Graph*
 * input(prompt + negative prompt) -> cliptokenizer(yes or no) -> clip(batch) ->
 * output(text_embeddings)
 */
dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *input,
                            const std::string &output,
                            base::InferenceType inference_type,
                            std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, input, output);

  /**
   * @brief
   * yes or no,
   * 目前这个阶段暂时没做tokenizer，所以input就是text_ids，会在内部做一个拷贝操作
   */
  dag::Node *tokenizer =
      graph->createNode<CLIPTokenizer>("CLIPTokenizer", input, "input_ids");
  tokenizer->setParam(param[0]);

  /**
   * @brief
   * batch = 2， 多batch的推理
   */
  dag::Infer *clip_infer =
      graph->create<Infer>("clip_infer", inference_type, "input_ids", output);
  clip_infer->setParam(param[1]);
}

enum SchedulerType : public int {
  kSchedulerDDIM = 0x0000,
  kSchedulerDPM,
  kSchedulerEulerA,
  kSchedulerLMSD,
  kSchedulerPNDM,
  kSchedulerNotSupport,
};

class NNDEPLOY_CC_API SchedulerParam : public base::Param {
 public:
  std::string version_ = "v2.1";
  int num_train_timesteps_ = 1000;
  float beta_start_ = 0.00085;
  float beta_end_ = 0.012;
  // v_prediction or epsilon
  std::string prediction_type = "v_prediction";
  int steps_offset_ = 1;
};

class NNDEPLOY_CC_API Scheduler : public Loop {
 public:
  Scheduler(const std::string &name, Edge *input, Edge *output)
      : Loop(name, input, output) {}
  Scheduler(const std::string &name, std::initializer_list<Edge *> inputs,
            std::initializer_list<Edge *> outputs)
      : Loop(name, inputs, outputs) {}
  virtual ~Scheduler() {}

 protected:
  SchedulerType scheduler_type_ = kSchedulerDDIM;
};

dag::Graph *createSchedulerUNetGraph(const std::string &name,
                                     const std::string &input,
                                     const std::string &output,
                                     SchedulerType scheduler_type,
                                     base::InferenceType inference_type,
                                     std::vector<base::Param *> &param) {
  dag::Scheduler *scheduler =
      createScheduler(name, input, output, scheduler_type);
  scheduler->setParam(param[0]);

  dag::Node *build_sample = scheduler->create<BuildSample>(
      "build_sample", inference_type, "", "sample");
  build_sample->setParam(param[1]);

  dag::Node *build_timestep = scheduler->create<BuildTimestep>(
      "build_timestep", inference_type, "", "timestep");
  build_sample->setParam(param[2]);

  dag::Node *copy = scheduler->create<Copy>("copy", inference_type, input,
                                            "encoder_hidden_states");
  build_sample->setParam(param[3]);

  dag::Infer *unet_infer = scheduler->create<Infer>(
      "unet_infer", inference_type,
      {"sample", "timestep", "encoder_hidden_states"}, output);
  build_sample->setParam(param[4]);
}

/**
 * @brief Create a Stable Diffusion Text2Image Graph object
 *
 * @param name
 * @param input
 * @param output
 * @param param
 * @return dag::Graph*
 * input(prompt + negative prompt) -> clip -> schedule_unet -> vae ->
 * output(image)
 */
dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string &name, dag::Edge *input, dag::Edge *output,
    base::InferenceType clip_inference_type,
    base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, input, output);

  // 边由谁来管理呢？
  dag::Graph *clip = createCLIPGraph("clip", input, "text_embeddings", param);
  graph->addNode(clip, false);

  std::vector<base::Param *>::iterator it = param.begin();
  // 边由谁来管理呢？
  std::vector<base::Param *> schedule_unet_param(it + 2, it + 4);
  dag::Graph *schedule_unet = createScheduleUNetGraph(
      "schedule_unet", "text_embeddings", "latent", schedule_unet_param);
  graph->addNode(schedule_unet, false);

  // 边由谁来管理呢？
  dag::Infer *vae_infer =
      graph->create<Infer>("vae_infer", vae_inference_type, "latent", output);
  build_sample->setParam(param[4]);

  return graph;
}

}  // namespace model
}  // namespace nndeploy