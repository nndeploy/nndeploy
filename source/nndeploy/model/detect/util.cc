
#include "nndeploy/model/detect/util.h"

namespace nndeploy {
namespace model {

std::array<float, 4> getOriginBox(float xmin, float ymin, float xmax,
                                  float ymax, const float* scale_factor,
                                  float x_offset, float y_offset, int ori_width,
                                  int ori_height) {
  xmin = std::max(xmin / scale_factor[0] + x_offset, 0.f);
  ymin = std::max(ymin / scale_factor[1] + y_offset, 0.f);
  xmax = std::min(xmax / scale_factor[2] + x_offset, (float)ori_width - 1.f);
  ymax = std::min(ymax / scale_factor[3] + y_offset, (float)ori_height - 1.f);
  return {xmin, ymin, xmax, ymax};
}

std::array<float, 4> getOriginBox(const std::array<float, 4>& box,
                                  const float* scale_factor, float x_offset,
                                  float y_offset, int ori_width,
                                  int ori_height) {
  float xmin = std::max(box[0] / scale_factor[0] + x_offset, 0.f);
  float ymin = std::max(box[1] / scale_factor[1] + y_offset, 0.f);
  float xmax =
      std::min(box[3] / scale_factor[2] + x_offset, (float)ori_width - 1.f);
  float ymax =
      std::min(box[4] / scale_factor[3] + y_offset, (float)ori_height - 1.f);
  return {xmin, ymin, xmax, ymax};
}

float computeIOU(float xmin0, float ymin0, float xmax0, float ymax0,
                 float xmin1, float ymin1, float xmax1, float ymax1) {
  const float area_i = (ymax0 - ymin0) * (xmax0 - xmin0);
  const float area_j = (ymax1 - ymin1) * (xmax1 - xmin1);
  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }
  const float intersection_x_min = std::max<float>(xmin0, xmin1);
  const float intersection_y_min = std::max<float>(ymin0, ymin1);
  const float intersection_x_max = std::min<float>(xmax0, xmax1);
  const float intersection_y_max = std::min<float>(ymax0, ymax1);
  const float intersection_area =
      std::max<float>(intersection_y_max - intersection_y_min, 0.0) *
      std::max<float>(intersection_x_max - intersection_x_min, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

float computeIOU(const std::array<float, 4>& box0,
                 const std::array<float, 4>& box1) {
  const float area_i = (box0[3] - box0[1]) * (box0[2] - box0[0]);
  const float area_j = (box1[3] - box1[1]) * (box1[2] - box1[0]);
  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }
  const float intersection_x_min = std::max<float>(box0[0], box1[0]);
  const float intersection_y_min = std::max<float>(box0[1], box1[1]);
  const float intersection_x_max = std::min<float>(box0[2], box1[2]);
  const float intersection_y_max = std::min<float>(box0[3], box1[3]);
  const float intersection_area =
      std::max<float>(intersection_y_max - intersection_y_min, 0.0) *
      std::max<float>(intersection_x_max - intersection_x_min, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

float computeIOU(const float* boxes, int i, int j) {
  const float x_min_i = std::min<float>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
  const float y_min_i = std::min<float>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
  const float x_max_i = std::max<float>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
  const float y_max_i = std::max<float>(boxes[i * 4 + 1], boxes[i * 4 + 3]);

  const float x_min_j = std::min<float>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
  const float y_min_j = std::min<float>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
  const float x_max_j = std::max<float>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
  const float y_max_j = std::max<float>(boxes[j * 4 + 1], boxes[j * 4 + 3]);

  const float area_i = (y_max_i - y_min_i) * (x_max_i - x_min_i);
  const float area_j = (y_max_j - y_min_j) * (x_max_j - x_min_j);
  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }
  const float intersection_x_min = std::max<float>(x_min_i, x_min_j);
  const float intersection_y_min = std::max<float>(y_min_i, y_min_j);
  const float intersection_x_max = std::min<float>(x_max_i, x_max_j);
  const float intersection_y_max = std::min<float>(y_max_i, y_max_j);
  const float intersection_area =
      std::max<float>(intersection_y_max - intersection_y_min, 0.0) *
      std::max<float>(intersection_x_max - intersection_x_min, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

base::Status computeNMS(const DetectResults& src, std::vector<int>& keep_idxs,
                 const float iou_threshold) {
  for (auto i = 0; i < src.result_.size(); ++i) {
    keep_idxs[i] = i;
  }
  for (auto i = 0; i < keep_idxs.size(); ++i) {
    auto n = keep_idxs[i];
    if (n < 0) {
      continue;
    }
    for (auto j = i + 1; j < keep_idxs.size(); ++j) {
      auto m = keep_idxs[j];
      if (m < 0) {
        continue;
      }
      float iou = computeIOU(src.result_[n].bbox_, src.result_[m].bbox_);

      if (iou > iou_threshold) {
        keep_idxs[j] = -1;
      }
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace pipeline
}  // namespace nndeploy
