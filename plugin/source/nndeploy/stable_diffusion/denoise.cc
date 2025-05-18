#include "nndeploy/stable_diffusion/denoise.h"

#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge/pipeline_edge.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_add.h"
#include "nndeploy/op/op_concat.h"
#include "nndeploy/op/op_muls.h"
#include "nndeploy/op/op_split.h"
#include "nndeploy/op/op_sub.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/scheduler.h"
#include "nndeploy/stable_diffusion/utils.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API DenoiseParam : public base::Param {
 public:
  DenoiseParam() = default;
  virtual ~DenoiseParam() {}

  DenoiseParam(const DDIMSchedulerParam *scheduler,
               const Text2ImageParam *text2img) {
    if (scheduler) {
      scheduler_param_ = *scheduler;
    }
    if (text2img) {
      text2image_param_ = *text2img;
    }
  }

  PARAM_COPY(DenoiseParam);
  PARAM_COPY_TO(DenoiseParam);

 public:
  DDIMSchedulerParam scheduler_param_;
  Text2ImageParam text2image_param_;
};

class NNDEPLOY_CC_API InitLatentsNode : public dag::Node {
 public:
  InitLatentsNode(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::InitLatentsNode";
    param_ = std::make_shared<DDIMSchedulerParam>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~InitLatentsNode() {}
  virtual base::Status init() { return base::kStatusCodeOk; }
  virtual base::Status deinit() { return base::kStatusCodeOk; }
  virtual base::Status run() {
    std::shared_ptr<DDIMSchedulerParam> scheduler_param =
        std::dynamic_pointer_cast<DDIMSchedulerParam>(param_);
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {1, scheduler_param->unet_channels_,
                           scheduler_param->image_height_ / 8,
                           scheduler_param->image_width_ / 8};
    device::Tensor *latents = this->getOutput(0)->create(device, latents_desc);

    std::mt19937 generator;
    initializeLatents(generator, scheduler_param->init_noise_sigma_, latents);

    this->getOutput(0)->notifyWritten(latents);

    index_++;

    return base::kStatusCodeOk;
  }

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  void setSize(int size) {
    if (size > 0) {
      size_ = size;
    }
  }
  int getSize() { return size_; }

 private:
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API DDIMScheduleNode : public dag::Node {
 public:
  DDIMScheduleNode(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}

  virtual ~DDIMScheduleNode() {}

  void setSchedulerParam(DDIMSchedulerParam *param) {
    scheduler_param_ = param;
  }
  void setScheduler(Scheduler *scheduler) { scheduler_ = scheduler; }

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;
    DDIMSchedulerParam *ddim_param = (DDIMSchedulerParam *)scheduler_param_;

    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc noise_desc;
    noise_desc.data_type_ = base::dataTypeOf<float>();
    noise_desc.data_format_ = base::kDataFormatNCHW;
    noise_desc.shape_ = {1, ddim_param->unet_channels_,
                         ddim_param->image_height_ / 8,
                         ddim_param->image_width_ / 8};
    noise_pred_ = new device::Tensor(device, noise_desc);
    noise_pred_uncond_ = new device::Tensor(device, noise_desc);
    noise_sub_ = new device::Tensor(device, noise_desc);
    noise_muls_ = new device::Tensor(device, noise_desc);

    device::TensorDesc scalar_desc;
    scalar_desc.data_type_ = base::dataTypeOf<float>();
    scalar_desc.data_format_ = base::kDataFormatNC;
    scalar_desc.shape_.emplace_back(1);
    scalar_ = new device::Tensor(device, scalar_desc);
    scalar_->set(ddim_param->guidance_scale_);

    do_classifier_free_guidance_ =
        (ddim_param->guidance_scale_ > 1.0) ? true : false;

    return status;
  }

  virtual base::Status deinit() {
    if (noise_pred_ != nullptr) {
      delete noise_pred_;
      noise_pred_ = nullptr;
    }
    if (noise_pred_uncond_ != nullptr) {
      delete noise_pred_uncond_;
      noise_pred_uncond_ = nullptr;
    }
    if (noise_sub_ != nullptr) {
      delete noise_sub_;
      noise_sub_ = nullptr;
    }
    if (noise_muls_ != nullptr) {
      delete noise_muls_;
      noise_muls_ = nullptr;
    }
    if (scalar_ != nullptr) {
      delete scalar_;
      scalar_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    DDIMSchedulerParam *ddim_param = (DDIMSchedulerParam *)scheduler_param_;
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latent_desc;
    latent_desc.data_type_ = base::dataTypeOf<float>();
    latent_desc.data_format_ = base::kDataFormatNCHW;
    latent_desc.shape_ = {1, ddim_param->unet_channels_,
                          ddim_param->image_height_ / 8,
                          ddim_param->image_width_ / 8};
    latents = this->getOutput(0)->create(device, latent_desc);

    device::Tensor *unet_output_t = this->getInput(0)->getTensor(this);
    if (do_classifier_free_guidance_) {
      std::vector<device::Tensor *> outputs = {noise_pred_uncond_, noise_pred_};
      std::shared_ptr<ir::SplitParam> split_param =
          std::make_shared<ir::SplitParam>();
      split_param->axis_ = 0;
      split_param->num_outputs_ = 2;
      op::split(unet_output_t, split_param, outputs);
      op::sub(noise_pred_, noise_pred_uncond_, noise_sub_);
      op::muls(scalar_, noise_sub_, noise_muls_);
      op::add(noise_pred_uncond_, noise_muls_, noise_pred_);
    } else {
      noise_pred_ = unet_output_t;
    }

    device::Tensor *prev_latents = this->getInput(1)->getTensor(this);
    device::Tensor *timestep_t = this->getInput(2)->getTensor(this);
    scheduler_->step(noise_pred_, timestep_t, prev_latents, latents);

    // this->getOutput(0)->notifyWritten(latents);

    return base::kStatusCodeOk;
  }

 private:
  Scheduler *scheduler_ = nullptr;
  SchedulerParam *scheduler_param_ = nullptr;

  bool do_classifier_free_guidance_ = false;

  device::Tensor *latent = nullptr;
  device::Tensor *latents = nullptr;
  device::Tensor *noise_pred_ = nullptr;
  device::Tensor *noise_pred_uncond_ = nullptr;
  device::Tensor *noise_sub_ = nullptr;
  device::Tensor *noise_muls_ = nullptr;
  device::Tensor *scalar_ = nullptr;
  device::Tensor *timestep_t = nullptr;
};

class NNDEPLOY_CC_API DenoiseNode : public dag::CompositeNode {
 public:
  DenoiseNode(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : CompositeNode(name, inputs, outputs),
        scheduler_type_(stable_diffusion::kSchedulerTypeDDIM) {
    key_ = "nndeploy::stable_diffusion::DenoiseNode";
    param_ = std::make_shared<DenoiseParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();

    scheduler_ = createScheduler(scheduler_type_);
  }
  virtual ~DenoiseNode() {
    if (scheduler_ != nullptr) {
      delete scheduler_;
      scheduler_ = nullptr;
    }
  }
  virtual base::Status make() {
    base::Status status = base::kStatusCodeOk;

    dag::NodeDesc infer_desc("nndeploy::infer::Infer", "infer",
                             {"text_embeddings", "cfg_latents", "timestep"},
                             {"unet_output"});
    infer_node_ = (infer::Infer *)this->createInfer<infer::Infer>(
        infer_desc, base::kInferenceTypeOnnxRuntime);

    dag::NodeDesc schedule_desc(
        "nndeploy::stable_diffusion::DDIMScheduleNode", "schedule",
        {"unet_output", "prev_latents", "timestep"}, {"latents"});
    schedule_node_ = (DDIMScheduleNode *)this->createNode(schedule_desc);

    return status;
  }

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;
    denoise_param_ = (DenoiseParam *)param_.get();
    scheduler_param_ = &denoise_param_->scheduler_param_;
    text2image_param_ = &denoise_param_->text2image_param_;

    if (scheduler_param_ == nullptr) {
      NNDEPLOY_LOGE("scheduler param is null!");
      return base::kStatusCodeErrorNullParam;
    }
    status = scheduler_->init(scheduler_param_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "scheduler init failed!");

    do_classifier_free_guidance_ =
        (scheduler_param_->guidance_scale_ > 1.0) ? true : false;

    device_ = device::getDefaultHostDevice();
    int chunk_num = do_classifier_free_guidance_ ? 2 : 1;

    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {chunk_num, scheduler_param_->unet_channels_,
                           scheduler_param_->image_height_ / 8,
                           scheduler_param_->image_width_ / 8};
    dag::Edge *cfg_latents_edge = this->getEdge("cfg_latents");
    cfg_latents_ = cfg_latents_edge->create(device_, latents_desc);

    device::TensorDesc timesteps_desc;
    timesteps_desc.data_type_ = base::dataTypeOf<float>();
    timesteps_desc.data_format_ = base::kDataFormatNC;
    timesteps_desc.shape_ = {1};
    dag::Edge *timestep_edge = this->getEdge("timestep");
    timestep_ = timestep_edge->create(device_, timesteps_desc);

    inference::InferenceParam *infer_param = new inference::InferenceParam();
    infer_param->device_type_ = text2image_param_->device_type_;
    infer_param->model_type_ = text2image_param_->model_type_;
    infer_param->is_path_ = text2image_param_->is_path_;
    std::vector<std::string> onnx_path = {text2image_param_->model_value_[2]};
    infer_param->model_value_ = onnx_path;
    infer_node_->setParam(infer_param);
    infer_node_->init();

    schedule_node_->setSchedulerParam(scheduler_param_);
    schedule_node_->setScheduler(scheduler_);
    schedule_node_->init();

    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    status = scheduler_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "scheduler deinit failed!");
    return status;
  }

  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;

    device::Tensor *init_latents = this->getInput(0)->getTensor(this);
    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {1, scheduler_param_->unet_channels_,
                           scheduler_param_->image_height_ / 8,
                           scheduler_param_->image_width_ / 8};
    dag::Edge *prev_latents_edge = this->getEdge("prev_latents");
    device::Tensor *prev_latents =
        prev_latents_edge->create(device_, latents_desc);
    dag::Edge *latents_edge = this->getEdge("latents");
    device::Tensor *latents = latents_edge->create(device_, latents_desc);

    init_latents->copyTo(prev_latents);

    scheduler_->setTimesteps();
    std::vector<int> timesteps = scheduler_->getTimesteps();
    ProgressBar progress(timesteps.size(), 100, "Denoise", "Processing...");
    int i = 0;
    infer_node_->updateInput();
    for (const auto &val : timesteps) {
      if (do_classifier_free_guidance_) {
        std::shared_ptr<ir::ConcatParam> param =
            std::make_shared<ir::ConcatParam>();
        param->axis_ = 0;
        op::concat({prev_latents, prev_latents}, param, cfg_latents_);
      } else {
        prev_latents->copyTo(cfg_latents_);
      }
      timestep_->set((float)val);

      infer_node_->run();
      schedule_node_->run();
      latents->copyTo(prev_latents);
      progress.update(i++);
    }
    progress.finish();
    device::Tensor *latents_out =
        this->getOutput(0)->create(device_, latents_desc);
    latents->copyTo(latents_out);
    this->getOutput(0)->notifyWritten(latents_out);

    setRunningFlag(false);
    return status;
  }

 private:
  SchedulerType scheduler_type_;
  Scheduler *scheduler_ = nullptr;  // 指向 DDIMScheduler
  DenoiseParam *denoise_param_ = nullptr;
  DDIMSchedulerParam *scheduler_param_ = nullptr;
  Text2ImageParam *text2image_param_ = nullptr;

  device::Device *device_ = nullptr;

  device::Tensor *cfg_latents_ = nullptr;
  device::Tensor *timestep_ = nullptr;

  bool do_classifier_free_guidance_ = false;

  infer::Infer *infer_node_ = nullptr;
  DDIMScheduleNode *schedule_node_ = nullptr;
};

dag::Graph *createDenoiseGraph(const std::string &name,
                               dag::Edge *text_embeddings, dag::Edge *latents,
                               SchedulerType scheduler_type,
                               base::InferenceType inference_type,
                               std::vector<base::Param *> &param, int iter) {
  Text2ImageParam *text2image_param = (Text2ImageParam *)(param[0]);
  DDIMSchedulerParam *scheduler_param = (DDIMSchedulerParam *)(param[1]);

  dag::Graph *denoise_graph =
      new dag::Graph(name, {text_embeddings}, {latents});

  dag::Edge *init_latents = denoise_graph->createEdge("init_latents");
  dag::NodeDesc init_latents_desc("nndeploy::stable_diffusion::InitLatentsNode",
                                  "init_latents", {},
                                  {init_latents->getName()});
  InitLatentsNode *init_latents_node =
      (InitLatentsNode *)denoise_graph->createNode(init_latents_desc);
  init_latents_node->setParam(scheduler_param);
  init_latents_node->setSize(iter);

  dag::NodeDesc denoise_desc(
      "nndeploy::stable_diffusion::DenoiseNode", "denoise",
      {init_latents->getName(), text_embeddings->getName()},
      {latents->getName()});
  DenoiseNode *denoise_node =
      (DenoiseNode *)denoise_graph->createNode(denoise_desc);
  auto denoise_param =
      std::make_shared<DenoiseParam>(scheduler_param, text2image_param);
  denoise_node->setParam(denoise_param.get());
  denoise_node->make();

  return denoise_graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::InitLatentsNode", InitLatentsNode);
REGISTER_NODE("nndeploy::stable_diffusion::DenoiseNode", DenoiseNode);
REGISTER_NODE("nndeploy::stable_diffusion::DDIMScheduleNode", DDIMScheduleNode);

}  // namespace stable_diffusion
}  // namespace nndeploy