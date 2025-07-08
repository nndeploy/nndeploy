#include "nndeploy/stable_diffusion/text2image.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#include "nndeploy/stable_diffusion/clip.h"
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
    key_ = "nndeploy::stable_diffusion::TokenParam";
    desc_ = "construct tokenize text [String => TokenizerText]";
    this->setOutputTypeInfo<tokenizer::TokenizerText>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }

  virtual ~InitTokenText() {}

  virtual base::Status run() {
    setRunningFlag(true);
    tokenizer::TokenizerText *prompt_text = new tokenizer::TokenizerText();
    prompt_text->texts_ = {prompt_};
    this->getOutput(0)->set(prompt_text);
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

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("prompt_", rapidjson::Value(prompt_.c_str(), allocator),
                   allocator);
    json.AddMember("size_", size_, allocator);
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
    if (json.HasMember("size_") && json["size_"].IsInt()) {
      int size = json["size_"].GetInt();
      if (size > 0) {
        this->setSize(size);
      }
    }
    return status;
  }

 private:
  std::string prompt_;
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API SaveImage : public dag::Node {
 public:
  SaveImage(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::SaveImage";
    desc_ = "save cvmat to image";
    this->setInputTypeInfo<device::Tensor>();
    node_type_ = dag::NodeType::kNodeTypeOutput;
  }

  virtual ~SaveImage() {}

  virtual base::Status run() {
    device::Tensor *input =
        (device::Tensor *)(this->getInput(0)->getTensor(this));
    float *ptr = (float *)(input->getData());
    int n = input->getBatch();
    int c = input->getChannel();
    int h = input->getHeight();
    int w = input->getWidth();
    std::vector<cv::Mat> images = ToImages(ptr, n, c, h, w);

    for (size_t i = 0; i < images.size(); ++i) {
      cv::imwrite(output_path_, images[i]);
      NNDEPLOY_LOGI("images has been saved to %s\n", output_path_.c_str());
    }
    return base::kStatusCodeOk;
  }

  void setOutputPath(const std::string &output_path) {
    output_path_ = output_path;
  }

  std::vector<cv::Mat> ToImages(float *images, int N, int C, int H, int W) {
    std::vector<cv::Mat> result;

    for (int n = 0; n < N; ++n) {
      cv::Mat img(H, W, (C == 1) ? CV_8UC1 : CV_8UC3);

      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            float val = (images[idx] + 1.0f) * 255.0f / 2.0f;
            val = std::min(std::max(val, 0.0f), 255.0f);
            uint8_t pixelValue = static_cast<uint8_t>(std::round(val));

            if (C == 1) {
              img.at<uint8_t>(h, w) = pixelValue;
            } else {
              img.at<cv::Vec3b>(h, w)[c] = pixelValue;
            }
          }
        }
      }

      if (C == 3) {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
      }

      result.push_back(img);
    }

    return result;
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("output_path_",
                   rapidjson::Value(output_path_.c_str(), allocator),
                   allocator);
    return status;
  }

  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("output_path_") && json["output_path_"].IsString()) {
      std::string output_path = json["output_path_"].GetString();
      this->setOutputPath(output_path);
    }
    return status;
  }

 private:
  std::string output_path_;
};

dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string name, std::string prompt_str,
    std::string negative_prompt_str, base::InferenceType inference_type,
    SchedulerType scheduler_type, std::vector<base::Param *> &param, int iter) {
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

  SaveImage *save_node = (SaveImage *)graph->createNode<SaveImage>(
      "save_image", std::vector<dag::Edge *>{output},
      std::vector<dag::Edge *>{});
  Text2ImageParam *text2image_param = (Text2ImageParam *)(param[0]);
  std::string output_path = text2image_param->output_path_;
  save_node->setOutputPath(output_path);

  return graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::InitTokenText", InitTokenText);
REGISTER_NODE("nndeploy::stable_diffusion::SaveImage", SaveImage);

}  // namespace stable_diffusion
}  // namespace nndeploy