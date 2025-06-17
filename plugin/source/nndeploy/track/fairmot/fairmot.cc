#include "nndeploy/track/fairmot/fairmot.h"

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
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"
#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace track {

base::Status FairMotPreProcess::run() {
  FairMotPreParam* tmp_param = dynamic_cast<FairMotPreParam*>(param_.get());
  cv::Mat* src = inputs_[0]->getCvMat(this);
  device::Device* device = device::getDefaultHostDevice();

  device::TensorDesc desc;
  desc.data_type_ = tmp_param->data_type_;
  desc.data_format_ = tmp_param->data_format_;
  if (desc.data_format_ == base::kDataFormatNCHW) {
    desc.shape_ = {
        1, preprocess::getChannelByPixelType(tmp_param->dst_pixel_type_),
        tmp_param->h_, tmp_param->w_};
  } else {
    desc.shape_ = {
        1, tmp_param->h_, tmp_param->w_,
        preprocess::getChannelByPixelType(tmp_param->dst_pixel_type_)};
  }
  device::Tensor* dst = outputs_[1]->create(device, desc);

  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();

  cv::Mat tmp_cvt;
  if (tmp_param->src_pixel_type_ != tmp_param->dst_pixel_type_) {
    base::CvtColorType cvt_type = base::calCvtColorType(
        tmp_param->src_pixel_type_, tmp_param->dst_pixel_type_);
    if (cvt_type == base::kCvtColorTypeNotSupport) {
      NNDEPLOY_LOGE("cvtColor type not support");
      return base::kStatusCodeErrorNotSupport;
    }
    int cv_cvt_type =
        preprocess::OpenCvConvert::convertFromCvtColorType(cvt_type);
    cv::cvtColor(*src, tmp_cvt, cv_cvt_type);
  } else {
    tmp_cvt = *src;
  }

  cv::Mat tmp_resize;
  if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
    int interp_type = preprocess::OpenCvConvert::convertFromInterpType(
        tmp_param->interp_type_);
    cv::resize(tmp_cvt, tmp_resize, cv::Size(w, h), 0.0, 0.0, interp_type);
  } else {
    tmp_resize = tmp_cvt;
  }

  preprocess::OpenCvConvert::convertToTensor(
      tmp_resize, dst, tmp_param->normalize_, tmp_param->scale_,
      tmp_param->mean_, tmp_param->std_);
  outputs_[1]->notifyWritten(dst);

  device::TensorDesc shape_desc;
  shape_desc.data_type_ = tmp_param->data_type_;
  shape_desc.data_format_ = base::kDataFormatNC;
  shape_desc.shape_ = {1, 2};
  device::Tensor* shape_tensor = outputs_[0]->create(device, shape_desc);
  float* shape_data = reinterpret_cast<float*>(shape_tensor->getData());
  shape_data[0] = (float)h;
  shape_data[1] = (float)w;
  outputs_[0]->notifyWritten(shape_tensor);

  device::TensorDesc scale_desc;
  scale_desc.data_type_ = tmp_param->data_type_;
  scale_desc.data_format_ = base::kDataFormatNC;
  scale_desc.shape_ = {1, 2};
  device::Tensor* scale_tensor = outputs_[2]->create(device, scale_desc);
  float* scale_data = reinterpret_cast<float*>(scale_tensor->getData());
  scale_data[0] = h * 1.0 / src->rows;
  scale_data[1] = w * 1.0 / src->cols;
  outputs_[2]->notifyWritten(scale_tensor);

  return base::kStatusCodeOk;
}

void FairMotPostProcess::FilterDets(const float conf_thresh,
                                    const cv::Mat& dets,
                                    std::vector<int>* index) {
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
    if (score > conf_thresh) {
      index->push_back(i);
    }
  }
}

base::Status FairMotPostProcess::init() {
  jdeTracker_ = std::make_shared<JDETracker>();
  return base::kStatusCodeOk;
}

base::Status FairMotPostProcess::deinit() { return base::kStatusCodeOk; }

base::Status FairMotPostProcess::run() {
  FairMotPostParam* param = (FairMotPostParam*)param_.get();
  float conf_thresh = param->conf_thresh_;
  float tracked_thresh = param->tracked_thresh_;
  float min_box_area = param->min_box_area_;

  auto* det_tensor = inputs_[0]->getTensor(this);
  auto* emb_tensor = inputs_[1]->getTensor(this);
  if (!det_tensor || !emb_tensor) {
    NNDEPLOY_LOGE("Invalid input tensors for FairMotPostProcess.");
    return base::kStatusCodeErrorInvalidValue;
  }

  float* bbox_data = reinterpret_cast<float*>(det_tensor->getData());
  float* emb_data = reinterpret_cast<float*>(emb_tensor->getData());

  auto bbox_shape = det_tensor->getShape();
  auto emb_shape = emb_tensor->getShape();

  int num_boxes = bbox_shape[0];
  int emb_dim = emb_shape[1];

  cv::Mat dets(num_boxes, 6, CV_32FC1, bbox_data);
  cv::Mat emb(num_boxes, emb_dim, CV_32FC1, emb_data);

  // Step 1: Filter by conf threshold
  std::vector<int> valid;
  FilterDets(conf_thresh, dets, &valid);

  cv::Mat new_dets, new_emb;
  for (int i = 0; i < valid.size(); ++i) {
    new_dets.push_back(dets.row(valid[i]));
    new_emb.push_back(emb.row(valid[i]));
  }

  std::vector<Track> tracks;
  jdeTracker_->update(new_dets, new_emb, &tracks);

  MOTResult* result = new MOTResult();
  if (tracks.empty()) {
    // fallback: 使用第一个 bbox 作为 dummy track
    std::array<int, 4> box = {
        int(dets.at<float>(0, 0)), int(dets.at<float>(0, 1)),
        int(dets.at<float>(0, 2)), int(dets.at<float>(0, 3))};
    result->boxes.push_back(box);
    result->ids.push_back(1);
    result->scores.push_back(dets.at<float>(0, 4));
  } else {
    for (auto& track : tracks) {
      if (track.score < tracked_thresh) {
        continue;
      }
      float w = track.ltrb[2] - track.ltrb[0];
      float h = track.ltrb[3] - track.ltrb[1];
      bool vertical = w / h > 1.6;
      float area = w * h;
      if (area > min_box_area && !vertical) {
        std::array<int, 4> box = {int(track.ltrb[0]), int(track.ltrb[1]),
                                  int(track.ltrb[2]), int(track.ltrb[3])};
        result->boxes.push_back(box);
        result->ids.push_back(track.id);
        result->scores.push_back(track.score);
      }
    }
  }

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

}  // namespace track
}  // namespace nndeploy
