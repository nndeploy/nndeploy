#include "nndeploy/stable_diffusion/text2image.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#include "nndeploy/stable_diffusion/clip.h"
#include "nndeploy/stable_diffusion/denoise.h"
#include "nndeploy/stable_diffusion/vae.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API SaveImageNode : public dag::Node {
 public:
  SaveImageNode(const std::string &name, std::vector<dag::Edge *> inputs,
                std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}

  virtual ~SaveImageNode() {}

  base::Status init() { return base::kStatusCodeOk; }

  base::Status deinit() { return base::kStatusCodeOk; }

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
      std::string name = "image_" + std::to_string(i) + ".png";
      cv::imwrite(name, images[i]);
      NNDEPLOY_LOGI("images has been saved to %s", name.c_str());
    }
    return base::kStatusCodeOk;
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
};

dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string name, dag::Edge *prompt, dag::Edge *negative_prompt,
    dag::Edge *output, base::InferenceType clip_inference_type,
    base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, SchedulerType scheduler_type,
    std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt}, {output});

  dag::Edge *text_embeddings = graph->createEdge("text_embeddings");
  dag::Graph *clip_graph =
      createCLIPGraph("clip", prompt, negative_prompt, text_embeddings,
                      clip_inference_type, param);
  graph->addNode(clip_graph, false);

  dag::Edge *latents = graph->createEdge("latents");
  dag::Graph *denoise_graph =
      createDenoiseGraph("denoise_ddim", text_embeddings, latents,
                         scheduler_type, unet_inference_type, param);
  graph->addNode(denoise_graph, false);

  dag::Node *vae_graph =
      createVAEGraph("vae", latents, output, vae_inference_type, param);
  graph->addNode(vae_graph, false);

  dag::Node *save_node = graph->createNode<SaveImageNode>(
      "save_node", std::vector<dag::Edge *>{output},
      std::vector<dag::Edge *>{});

  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy