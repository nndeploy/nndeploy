#include "nndeploy/stable_diffusion/text2image.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#include "nndeploy/codec/codec.h"
#include "nndeploy/stable_diffusion/clip.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/denoise.h"
#include "nndeploy/stable_diffusion/vae.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API InitTokenText : public dag::Node {
 public:
  InitTokenText(const std::string &name, std::vector<dag::Edge *> inputs,
                std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::InitTokenText";
    desc_ = "Create TokenizerText from input prompt string.";
    this->setOutputTypeInfo<tokenizer::TokenizerText>();
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setIoType(dag::IOType::kIOTypeString);
  }

  virtual ~InitTokenText() {}

  virtual base::Status run() {
    setRunningFlag(true);
    tokenizer::TokenizerText *prompt_text = new tokenizer::TokenizerText();
    prompt_text->texts_ = {prompt_};
    this->getOutput(0)->set(prompt_text, false);
    this->getOutput(0)->notifyWritten(prompt_text);
    index_++;
    setRunningFlag(false);
    return base::kStatusCodeOk;
  }

  void setPrompt(std::string prompt) { prompt_ = prompt; }

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

  void setLoopCount(int loop_count) {
    if (loop_count > 0) {
      loop_count_ = loop_count;
      size_ = loop_count;
    }
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    this->addRequiredParam("prompt_");
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("prompt_", rapidjson::Value(prompt_.c_str(), allocator),
                   allocator);
    return status;
  }

  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("prompt_") && json["prompt_"].IsString()) {
      std::string prompt = json["prompt_"].GetString();
      this->setPrompt(prompt);
    }
    return status;
  }

 private:
  std::string prompt_;
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API TensorToMat : public dag::Node {
 public:
  TensorToMat(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::TensorToMat";
    desc_ = "Convert float tensor to cv::Mat image.";
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }

  virtual ~TensorToMat() {}

  virtual base::Status run() {
    auto *input = getInput(0)->getTensor(this);
    if (!input || !input->getData() ||
        input->getDataType() != base::dataTypeOf<float>() ||
        !device::isHostDeviceType(input->getDeviceType()) || input->getShape().size() != 4 ||
        (input->getDataFormat() != base::kDataFormatNCHW &&
         input->getDataFormat() != base::kDataFormatNHWC)) {
      return base::kStatusCodeErrorInvalidValue;
    }
    const int channels = input->getChannel();
    const int height = input->getHeight();
    const int width = input->getWidth();
    if (input->getBatch() != 1 || (channels != 1 && channels != 3) ||
        height <= 0 || width <= 0 ||
        static_cast<size_t>(height) > std::min(input->getSize(), input->getRealSize()) /
                                          sizeof(float) / channels / width) {
      return base::kStatusCodeErrorInvalidValue;
    }
    if (!input->isContinue()) return base::kStatusCodeErrorInvalidValue;
    const size_t pixels = static_cast<size_t>(height) * width;
    const auto *src = static_cast<const float *>(input->getData());
    const bool planar = input->getDataFormat() == base::kDataFormatNCHW;
    auto *image = new cv::Mat(height, width, CV_8UC(channels));
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      for (int c = 0; c < channels; ++c) {
        float val = src[planar ? c * pixels + pixel : pixel * channels + c];
        val = std::min(std::max((val + 1.0f) * 127.5f, 0.0f), 255.0f);
        image->data[pixel * channels + channels - 1 - c] = static_cast<uint8_t>(std::round(val));
      }
    }
    return getOutput(0)->set(image, false);
  }
};

dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string name, std::string prompt_str,
    std::string negative_prompt_str, base::InferenceType inference_type,
    SchedulerType scheduler_type, std::vector<base::Param *> &param, int iter) {
  auto *text2image_param = getText2ImageParam(param, 0, 3);
  if (!text2image_param || param.size() < 2 || !dynamic_cast<DDIMSchedulerParam *>(param[1])) {
    NNDEPLOY_LOGE("Expected tokenizer, CLIP, UNet and VAE models and DDIM parameters.\n");
    return nullptr;
  }
  dag::Graph *graph = new dag::Graph(name, {}, {});

  dag::Edge *prompt = graph->createEdge("prompt");
  dag::Edge *negative_prompt = graph->createEdge("negative_prompt");
  InitTokenText *init_node = (InitTokenText *)graph->createNode<InitTokenText>(
      "init_prompt", std::vector<dag::Edge *>{},
      std::vector<dag::Edge *>{prompt});
  init_node->setPrompt(prompt_str);
  InitTokenText *init_negative_node =
      (InitTokenText *)graph->createNode<InitTokenText>(
          "init_negative_prompt", std::vector<dag::Edge *>{},
          std::vector<dag::Edge *>{negative_prompt});
  init_negative_node->setPrompt(negative_prompt_str);

  dag::Edge *text_embeddings = graph->createEdge("text_embeddings");
  dag::Graph *clip_graph = createCLIPGraph(
      "clip", prompt, negative_prompt, text_embeddings, inference_type, param);
  graph->addNode(clip_graph, false);

  dag::Edge *latents = graph->createEdge("denoise_latents");
  dag::Graph *denoise_graph =
      createDenoiseGraph("denoise_ddim", text_embeddings, latents,
                         scheduler_type, inference_type, param, iter);
  graph->addNode(denoise_graph, false);

  dag::Edge *output = graph->createEdge("output");
  dag::Graph *vae_graph =
      createVAEGraph("vae", latents, output, inference_type, param);
  graph->addNode(vae_graph, false);

  dag::Edge *encode_input = graph->createEdge("encode_input");
  TensorToMat *convert_node = (TensorToMat *)graph->createNode<TensorToMat>(
      "tensor_to_mat", std::vector<dag::Edge *>{output},
      std::vector<dag::Edge *>{encode_input});

  std::string output_path = text2image_param->output_path_;
  codec::Encode *encode_node =
      codec::createEncode(base::kCodecTypeOpenCV, base::kCodecFlagImage,
                          "encode_node", encode_input);
  encode_node->setPath(output_path);
  graph->addNode(encode_node, false);

  return graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::InitTokenText", InitTokenText);
REGISTER_NODE("nndeploy::stable_diffusion::TensorToMat", TensorToMat);

}  // namespace stable_diffusion
}  // namespace nndeploy