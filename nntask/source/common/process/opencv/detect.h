#ifndef _NNTASK_SOURCE_COMMON_PROCESS_0PENCV_DETECT_H_
#define _NNTASK_SOURCE_COMMON_PROCESS_0PENCV_DETECT_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nntask/source/common/execution.h"
#include "nntask/source/common/opencv_include.h"
#include "nntask/source/common/packet.h"
#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

typedef struct {
  int width;
  int height;
} YoloSize;

typedef struct {
  std::string name;
  int stride;
  std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;

class BoxInfo {
 public:
  int x1, y1, x2, y2, label, id;
  float score;
};

class Yolo : public Execution {
 public:
  Yolo(nndeploy::base::DeviceType device_type, const std::string& name = "");
  virtual ~Yolo();

  virtual nndeploy::base::Status run() {
    cv::Mat* src = input_->getCvMat();
    nndeploy::device::Tensor* dst = output_->getTensor();

    int c = dst->getShapeIndex[1];
    int h = dst->getShapeIndex[2];
    int w = dst->getShapeIndex[3];

    cv::Mat tmp;
    cv::resize(*src, tmp, cv::Size(w, h));

    tmp.convertTo(tmp, CV_32FC3);
    tmp = tmp / 255.0f;

    std::vector<cv::Mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      float* data = (float*)dst->getPtr() + w * h * i;
      cv::Mat tmp(cv::Size(w, h), CV_32FC1, data);
    }

    cv::split(tmp, tmp_vec);
  }

 private:
  std::vector<BoxInfo> decode_infer(
      MNN::Tensor& data, int stride, const yolocv::YoloSize& frame_size,
      int net_size, int num_classes,
      const std::vector<yolocv::YoloSize>& anchors, float threshold) {
    std::vector<BoxInfo> result;
    int batchs, channels, height, width, pred_item;
    batchs = data.shape()[0];
    channels = data.shape()[1];
    height = data.shape()[2];
    width = data.shape()[3];
    pred_item = data.shape()[4];

    auto data_ptr = data.host<float>();
    for (int bi = 0; bi < batchs; bi++) {
      auto batch_ptr = data_ptr + bi * (channels * height * width * pred_item);
      for (int ci = 0; ci < channels; ci++) {
        auto channel_ptr = batch_ptr + ci * (height * width * pred_item);
        for (int hi = 0; hi < height; hi++) {
          auto height_ptr = channel_ptr + hi * (width * pred_item);
          for (int wi = 0; wi < width; wi++) {
            auto width_ptr = height_ptr + wi * pred_item;
            auto cls_ptr = width_ptr + 5;

            auto confidence = sigmoid(width_ptr[4]);

            for (int cls_id = 0; cls_id < num_classes; cls_id++) {
              float score = sigmoid(cls_ptr[cls_id]) * confidence;
              if (score > threshold) {
                float cx =
                    (sigmoid(width_ptr[0]) * 2.f - 0.5f + wi) * (float)stride;
                float cy =
                    (sigmoid(width_ptr[1]) * 2.f - 0.5f + hi) * (float)stride;
                float w =
                    pow(sigmoid(width_ptr[2]) * 2.f, 2) * anchors[ci].width;
                float h =
                    pow(sigmoid(width_ptr[3]) * 2.f, 2) * anchors[ci].height;

                BoxInfo box;

                box.x1 = std::max(
                    0, std::min(frame_size.width, int((cx - w / 2.f))));
                box.y1 = std::max(
                    0, std::min(frame_size.height, int((cy - h / 2.f))));
                box.x2 = std::max(
                    0, std::min(frame_size.width, int((cx + w / 2.f))));
                box.y2 = std::max(
                    0, std::min(frame_size.height, int((cy + h / 2.f))));
                box.score = score;
                box.label = cls_id;
                result.push_back(box);
              }
            }
          }
        }
      }
    }

    return result;
  }

  void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
      vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
                 (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
      for (int j = i + 1; j < int(input_boxes.size());) {
        float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
        float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
        float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
        float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
        float w = std::max(float(0), xx2 - xx1 + 1);
        float h = std::max(float(0), yy2 - yy1 + 1);
        float inter = w * h;
        float ovr = inter / (vArea[i] + vArea[j] - inter);
        if (ovr >= NMS_THRESH) {
          input_boxes.erase(input_boxes.begin() + j);
          vArea.erase(vArea.begin() + j);
        } else {
          j++;
        }
      }
    }
  }

  void scale_coords(std::vector<BoxInfo>& boxes, int w_from, int h_from,
                    int w_to, int h_to) {
    float w_ratio = float(w_to) / float(w_from);
    float h_ratio = float(h_to) / float(h_from);

    for (auto& box : boxes) {
      box.x1 *= w_ratio;
      box.x2 *= w_ratio;
      box.y1 *= h_ratio;
      box.y2 *= h_ratio;
    }
    return;
  }
};

}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_COMMON_PROCESS_0PENCV_DETECT_H_ */
