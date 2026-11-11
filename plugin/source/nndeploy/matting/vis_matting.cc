#include "nndeploy/matting/vis_matting.h"

namespace nndeploy {
namespace matting {

base::Status VisMatting::run() {
  if (inputs_.size() != 2 || outputs_.empty()) return base::kStatusCodeErrorInvalidParam;
  auto *img = inputs_[0]->getCvMat(this);
  auto *result = dynamic_cast<MattingResult *>(inputs_[1]->getParam(this));
  if (!img || img->empty() || !result || result->shape.size() != 2)
    return base::kStatusCodeErrorInvalidParam;
  auto h = result->shape[0], w = result->shape[1];
  if (h <= 0 || w <= 0 || h > INT_MAX || w > INT_MAX ||
      static_cast<uint64_t>(h * w) != result->alpha.size())
    return base::kStatusCodeErrorInvalidParam;

  cv::Mat pixels;
  img->convertTo(pixels, CV_8U);
  auto vis = std::make_unique<cv::Mat>();
  switch (pixels.channels()) {
    case 1: cv::cvtColor(pixels, *vis, cv::COLOR_GRAY2BGR); break;
    case 3: *vis = pixels; break;
    case 4: cv::cvtColor(pixels, *vis, cv::COLOR_BGRA2BGR); break;
    default: return base::kStatusCodeErrorNotSupport;
  }
  cv::Mat alpha(h, w, CV_32FC1, result->alpha.data());
  if (alpha.size() != vis->size()) cv::resize(alpha, alpha, vis->size());
  const cv::Vec3f background(153.f, 255.f, 120.f);
  for (int y = 0; y < vis->rows; ++y) {
    auto *row = vis->ptr<cv::Vec3b>(y);
    const auto *weights = alpha.ptr<float>(y);
    for (int x = 0; x < vis->cols; ++x) {
      float a = weights[x];
      row[x] = cv::Vec3b(cv::Vec3f(row[x]) * a + background * (1.f - a));
    }
  }
  auto status = outputs_[0]->set(vis.get(), false);
  if (status == base::kStatusCodeOk) vis.release();
  return status;
}

REGISTER_NODE("nndeploy::matting::VisMatting", VisMatting);

}  // namespace matting
}  // namespace nndeploy