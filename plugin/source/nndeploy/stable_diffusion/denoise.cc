#include "nndeploy/stable_diffusion/denoise.h"

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

class NNDEPLOY_CC_API DenoiseNode : public dag::Node {
 public:
  DenoiseNode(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs),
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

    cfg_latents_edge_ = new dag::Edge("cfg_latents");
    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {chunk_num, scheduler_param_->unet_channels_,
                           scheduler_param_->image_height_ / 8,
                           scheduler_param_->image_width_ / 8};
    cfg_latents_ = cfg_latents_edge_->create(device_, latents_desc);

    timestep_edge_ = new dag::Edge("timestep");
    device::TensorDesc timesteps_desc;
    timesteps_desc.data_type_ = base::dataTypeOf<float>();
    timesteps_desc.data_format_ = base::kDataFormatNC;
    timesteps_desc.shape_ = {1};
    timestep_ = timestep_edge_->create(device_, timesteps_desc);

    text_embeddings_edge_ = new dag::Edge("text_embeddings_copy");
    unet_output_edge_ = new dag::Edge("unet_output");
    prev_latents_edge_ = new dag::Edge("prev_latents");
    latents_edge_ = new dag::Edge("latents");

    infer_node_ = new infer::Infer(
        "infer", {text_embeddings_edge_, cfg_latents_edge_, timestep_edge_},
        {unet_output_edge_}, base::kInferenceTypeOnnxRuntime);
    inference::InferenceParam *infer_param = new inference::InferenceParam();
    infer_param->device_type_ = text2image_param_->device_type_;
    infer_param->model_type_ = text2image_param_->model_type_;
    infer_param->is_path_ = text2image_param_->is_path_;
    std::vector<std::string> onnx_path = {text2image_param_->model_value_[2]};
    infer_param->model_value_ = onnx_path;
    infer_node_->setParam(infer_param);
    infer_node_->init();

    schedule_node_ = new DDIMScheduleNode(
        "schedule", {unet_output_edge_, prev_latents_edge_, timestep_edge_},
        {latents_edge_});
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
    if (cfg_latents_ != nullptr) {
      delete cfg_latents_;
      cfg_latents_ = nullptr;
    }
    if (timestep_ != nullptr) {
      delete timestep_;
      timestep_ = nullptr;
    }
    if (text_embeddings_edge_ != nullptr) {
      delete text_embeddings_edge_;
      text_embeddings_edge_ = nullptr;
    }
    if (unet_output_edge_ != nullptr) {
      delete unet_output_edge_;
      unet_output_edge_ = nullptr;
    }
    if (prev_latents_edge_ != nullptr) {
      delete prev_latents_edge_;
      prev_latents_edge_ = nullptr;
    }
    if (latents_edge_ != nullptr) {
      delete latents_edge_;
      latents_edge_ = nullptr;
    }
    if (infer_node_ != nullptr) {
      delete infer_node_;
      infer_node_ = nullptr;
    }
    if (schedule_node_ != nullptr) {
      delete schedule_node_;
      schedule_node_ = nullptr;
    }
    return status;
  }
  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;

    device::TensorDesc text_embeddings_desc =
        this->getInput(1)->getTensor(this)->getDesc();
    device::Tensor *text_embeddings_copy =
        text_embeddings_edge_->create(device_, text_embeddings_desc);
    this->getInput(1)->getTensor(this)->copyTo(text_embeddings_copy);

    device::Tensor *init_latents = this->getInput(0)->getTensor(this);

    // void *data_ptr = init_latents->getData();
    // float *float_data = static_cast<float *>(data_ptr);
    // std::cout << "init_latents: " << float_data[0] << std::endl;

    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {1, scheduler_param_->unet_channels_,
                           scheduler_param_->image_height_ / 8,
                           scheduler_param_->image_width_ / 8};
    device::Tensor *prev_latents =
        prev_latents_edge_->create(device_, latents_desc);
    device::Tensor *latents = latents_edge_->create(device_, latents_desc);

    init_latents->copyTo(prev_latents);

    scheduler_->setTimesteps();
    std::vector<int> timesteps = scheduler_->getTimesteps();
    ProgressBar progress(timesteps.size(), 100, "Denoise", "Processing...");
    int i = 0;
    // NNDEPLOY_LOGE("Denoise node start loop!\n");
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

  dag::Edge *text_embeddings_edge_ = nullptr;
  dag::Edge *cfg_latents_edge_ = nullptr;
  dag::Edge *timestep_edge_ = nullptr;
  dag::Edge *unet_output_edge_ = nullptr;
  dag::Edge *prev_latents_edge_ = nullptr;
  dag::Edge *latents_edge_ = nullptr;

  bool do_classifier_free_guidance_ = false;

  infer::Infer *infer_node_ = nullptr;
  DDIMScheduleNode *schedule_node_ = nullptr;
};

dag::Graph *createDenoiseGraph(const std::string &name,
                               dag::Edge *text_embeddings, dag::Edge *latents,
                               SchedulerType scheduler_type,
                               base::InferenceType inference_type,
                               std::vector<base::Param *> &param) {
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

  dag::NodeDesc denoise_desc(
      "nndeploy::stable_diffusion::DenoiseNode", "denoise",
      {init_latents->getName(), text_embeddings->getName()},
      {latents->getName()});
  DenoiseNode *denoise_node =
      (DenoiseNode *)denoise_graph->createNode(denoise_desc);
  auto denoise_param =
      std::make_shared<DenoiseParam>(scheduler_param, text2image_param);
  denoise_node->setParam(denoise_param.get());

  // dag::Edge *init_latents = denoise_graph->createEdge("init_latents");
  // InitLatentsNode *init_latents_node =
  //     (InitLatentsNode *)denoise_graph->createNode<InitLatentsNode>(
  //         "init_latents", std::vector<dag::Edge *>{},
  //         std::vector<dag::Edge *>{init_latents});
  // init_latents_node->setParam(scheduler_param);

  // auto denoise_param =
  //     std::make_shared<DenoiseParam>(scheduler_param, text2image_param);
  // DenoiseNode *denoise_node =
  //     (DenoiseNode *)denoise_graph->createNode<DenoiseNode>(
  //         "denoise", std::vector<dag::Edge *>{init_latents, text_embeddings},
  //         std::vector<dag::Edge *>{latents});
  // denoise_node->setParam(denoise_param.get());

  return denoise_graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::InitLatentsNode", InitLatentsNode);
REGISTER_NODE("nndeploy::stable_diffusion::DenoiseNode", DenoiseNode);

}  // namespace stable_diffusion
}  // namespace nndeploy