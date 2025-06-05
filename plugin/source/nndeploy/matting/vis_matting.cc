#include "nndeploy/matting/vis_matting.h"

namespace nndeploy {
namespace matting {

base::Status VisMattingNode::run() {
  cv::Mat *img = inputs_[0]->getCvMat(this);
  MattingResult *result = (MattingResult *)inputs_[1]->getParam(this);

  cv::Mat *vis_img = new cv::Mat();
  img->copyTo(*vis_img);
  int channel = img->channels();
  int out_h = static_cast<int>(result->shape[0]);
  int out_w = static_cast<int>(result->shape[1]);
  int height = img->rows;
  int width = img->cols;

  std::vector<float> alpha_copy;
  alpha_copy.assign(result->alpha.begin(), result->alpha.end());
  float *alpha_ptr = static_cast<float *>(alpha_copy.data());
  cv::Mat alpha(out_h, out_w, CV_32FC1, alpha_ptr);

  if ((out_h != height) || (out_w != width)) {
    cv::resize(alpha, alpha, cv::Size(width, height));
  }

  if ((vis_img)->type() != CV_8UC3) {
    (vis_img)->convertTo((*vis_img), CV_8UC3);
  }

  uchar *vis_data = static_cast<uchar *>(vis_img->data);
  uchar *img_data = static_cast<uchar *>(img->data);
  float *alpha_data = reinterpret_cast<float *>(alpha.data);

  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      float alpha_val = alpha_data[i * width + j];
      vis_data[i * width * channel + j * channel + 0] =
          cv::saturate_cast<uchar>(
              static_cast<float>(img_data[i * width * 3 + j * 3 + 0]) *
                  alpha_val +
              (1.f - alpha_val) * 153.f);
      vis_data[i * width * channel + j * channel + 1] =
          cv::saturate_cast<uchar>(
              static_cast<float>(img_data[i * width * 3 + j * 3 + 1]) *
                  alpha_val +
              (1.f - alpha_val) * 255.f);
      vis_data[i * width * channel + j * channel + 2] =
          cv::saturate_cast<uchar>(
              static_cast<float>(img_data[i * width * 3 + j * 3 + 2]) *
                  alpha_val +
              (1.f - alpha_val) * 120.f);
    }
  }

  outputs_[0]->set(vis_img, false);

  return base::kStatusCodeOk;
}

}  // namespace matting
}  // namespace nndeploy