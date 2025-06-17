#include "nndeploy/detect/yolo/yolox.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"

namespace nndeploy {
namespace detect {

struct YOLOXAnchor {
  int grid0;
  int grid1;
  int stride;
};

void GenerateYOLOXAnchors(const std::vector<int> &size,
                          const std::vector<int> &downsample_strides,
                          std::vector<YOLOXAnchor> *anchors) {
  // size: tuple of input (width, height)
  // downsample_strides: downsample strides in YOLOX, e.g (8,16,32)
  const int width = size[0];
  const int height = size[1];
  for (const auto &ds : downsample_strides) {
    int num_grid_w = width / ds;
    int num_grid_h = height / ds;
    for (int g1 = 0; g1 < num_grid_h; ++g1) {
      for (int g0 = 0; g0 < num_grid_w; ++g0) {
        (*anchors).emplace_back(YOLOXAnchor{g0, g1, ds});
      }
    }
  }
}

base::Status YoloXPostProcess::run() {
  YoloXPostParam *param = (YoloXPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;
  std::vector<int> downsample_strides = {8, 16, 32};
  std::vector<int> model_size = {param->model_w_, param->model_h_};

  // 构建 YOLOX anchors
  std::vector<YOLOXAnchor> anchors;
  GenerateYOLOXAnchors(model_size, downsample_strides, &anchors);

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int num_boxes = tensor->getShapeIndex(1);  // anchor 数
  int box_dim = tensor->getShapeIndex(
      2);  // 通常是85: [dx, dy, dw, dh, obj, class_probs...]

  DetectResult *results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    float *data_batch = data + b * num_boxes * box_dim;
    DetectResult results_batch;

    for (int i = 0; i < num_boxes; ++i) {
      float *data_row = data_batch + i * box_dim;

      // 解码
      const YOLOXAnchor &anchor = anchors[i];
      float dx = data_row[0];
      float dy = data_row[1];
      float dw = data_row[2];
      float dh = data_row[3];
      float obj_conf = data_row[4];

      float grid0 = static_cast<float>(anchor.grid0);
      float grid1 = static_cast<float>(anchor.grid1);
      float stride = static_cast<float>(anchor.stride);

      float x_center = (dx + grid0) * stride;
      float y_center = (dy + grid1) * stride;
      float w = std::exp(dw) * stride;
      float h = std::exp(dh) * stride;

      float x0 = x_center - w * 0.5f;
      float y0 = y_center - h * 0.5f;
      float x1 = x_center + w * 0.5f;
      float y1 = y_center + h * 0.5f;

      for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        float score = obj_conf * data_row[5 + class_idx];
        if (score > score_threshold) {
          DetectBBoxResult bbox;
          bbox.index_ = b;
          bbox.label_id_ = class_idx;
          bbox.score_ = score;
          bbox.bbox_[0] = std::max(x0, 0.0f);
          bbox.bbox_[1] = std::max(y0, 0.0f);
          bbox.bbox_[2] = std::min(x1, static_cast<float>(param->model_w_));
          bbox.bbox_[3] = std::min(y1, static_cast<float>(param->model_h_));
          results_batch.bboxs_.emplace_back(bbox);
        }
      }
    }

    // NMS
    std::vector<int> keep_idxs(results_batch.bboxs_.size());
    computeNMS(results_batch, keep_idxs, param->nms_threshold_);
    for (auto i = 0; i < keep_idxs.size(); ++i) {
      auto n = keep_idxs[i];
      if (n < 0) continue;
      // 可选：归一化
      results_batch.bboxs_[n].bbox_[0] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[1] /= param->model_h_;
      results_batch.bboxs_[n].bbox_[2] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[3] /= param->model_h_;
      results->bboxs_.emplace_back(results_batch.bboxs_[n]);
    }
  }

  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}

// base::Status YoloXPostProcess::run() {
//   YoloXPostParam *param = (YoloXPostParam *)param_.get();
//   float score_threshold = param->score_threshold_;
//   int num_classes = param->num_classes_;

//   device::Tensor *tensor = inputs_[0]->getTensor(this);
//   float *data = (float *)tensor->getData();
//   int batch = tensor->getShapeIndex(0);
//   int height = tensor->getShapeIndex(1);
//   int width = tensor->getShapeIndex(2);

//   DetectResult *results = new DetectResult();

//   for (int b = 0; b < batch; ++b) {
//     // NNDEPLOY_LOGE("bk\n");
//     float *data_batch = data + b * height * width;
//     DetectResult results_batch;
//     for (int h = 0; h < height; ++h) {
//       float *data_row = data_batch + h * width;
//       float x_center = data_row[0];
//       float y_center = data_row[1];
//       float object_w = data_row[2];
//       float object_h = data_row[3];
//       float x0 = x_center - object_w * 0.5f;
//       x0 = x0 > 0.0 ? x0 : 0.0;
//       float y0 = y_center - object_h * 0.5f;
//       y0 = y0 > 0.0 ? y0 : 0.0;
//       float x1 = x_center + object_w * 0.5f;
//       x1 = x1 < param->model_w_ ? x1 : param->model_w_;
//       float y1 = y_center + object_h * 0.5f;
//       y1 = y1 < param->model_h_ ? y1 : param->model_h_;
//       float box_objectness = data_row[4];
//       for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
//         float score = box_objectness * data_row[5 + class_idx];
//         if (score > score_threshold) {
//           DetectBBoxResult bbox;
//           bbox.index_ = b;
//           bbox.label_id_ = class_idx;
//           bbox.score_ = score;
//           bbox.bbox_[0] = x0;
//           bbox.bbox_[1] = y0;
//           bbox.bbox_[2] = x1;
//           bbox.bbox_[3] = y1;
//           // NNDEPLOY_LOGE("score:%f, x0:%f, y0:%f, x1:%f, y1:%f\n", score,
//           x0,
//           // y0,
//           //               x1, y1);
//           results_batch.bboxs_.emplace_back(bbox);
//         }
//       }
//     }
//     std::vector<int> keep_idxs(results_batch.bboxs_.size());
//     computeNMS(results_batch, keep_idxs, param->nms_threshold_);
//     for (auto i = 0; i < keep_idxs.size(); ++i) {
//       auto n = keep_idxs[i];
//       if (n < 0) {
//         continue;
//       }
//       results_batch.bboxs_[n].bbox_[0] /= param->model_w_;
//       results_batch.bboxs_[n].bbox_[1] /= param->model_h_;
//       results_batch.bboxs_[n].bbox_[2] /= param->model_w_;
//       results_batch.bboxs_[n].bbox_[3] /= param->model_h_;
//       results->bboxs_.emplace_back(results_batch.bboxs_[n]);
//     }
//   }
//   outputs_[0]->set(results, false);
//   return base::kStatusCodeOk;
// }

}  // namespace detect
}  // namespace nndeploy
