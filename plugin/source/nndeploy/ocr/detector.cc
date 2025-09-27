#include "nndeploy/ocr/detector.h"

#include "nndeploy/ocr/result.h"
#include "nndeploy/preprocess/opencv_convert.h"
#include "nndeploy/preprocess/opencv_util.h"
#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace ocr {

base::Status DetectorPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("version_", version_, allocator);
  json.AddMember("det_db_thresh_", det_db_thresh_, allocator);
  json.AddMember("det_db_box_thresh_", det_db_box_thresh_, allocator);
  json.AddMember("det_db_unclip_ratio_", det_db_unclip_ratio_, allocator);
  json.AddMember("det_db_score_mode_",
                 rapidjson::Value(det_db_score_mode_.c_str(), allocator),
                 allocator);
  json.AddMember("use_dilation_", use_dilation_, allocator);

  return base::kStatusCodeOk;
}

base::Status DetectorPostParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }

  if (json.HasMember("det_db_thresh_") && json["det_db_thresh_"].IsFloat()) {
    det_db_thresh_ = json["det_db_thresh_"].GetFloat();
  }
  if (json.HasMember("det_db_box_thresh_") &&
      json["det_db_box_thresh_"].IsFloat()) {
    det_db_box_thresh_ = json["det_db_box_thresh_"].GetFloat();
  }
  if (json.HasMember("det_db_unclip_ratio_") &&
      json["det_db_unclip_ratio_"].IsFloat()) {
    det_db_unclip_ratio_ = json["det_db_unclip_ratio_"].GetFloat();
  }
  if (json.HasMember("det_db_score_mode_") &&
      json["det_db_score_mode_"].IsString()) {
    det_db_score_mode_ = json["det_db_score_mode_"].GetString();
  }
  if (json.HasMember("use_dilation_") && json["use_dilation_"].IsBool()) {
    use_dilation_ = json["use_dilation_"].GetBool();
  }

  return base::kStatusCodeOk;
}

base::Status DetectorPreProcess::run() {
  DetectorPreProcessParam *tmp_param =
      dynamic_cast<DetectorPreProcessParam *>(param_.get());
  cv::Mat *src = inputs_[0]->getCvMat(this);
  device::Device *device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = tmp_param->data_type_;
  desc.data_format_ = tmp_param->data_format_;

  int origin_h = src->rows;
  int origin_w = src->cols;

  float ratio = 1.f;
  int max_wh = origin_w >= origin_h ? origin_w : origin_h;
  if (max_wh > tmp_param->max_side_len_) {
    if (origin_h > origin_w) {
      ratio = float(tmp_param->max_side_len_) / float(origin_h);
    } else {
      ratio = float(tmp_param->max_side_len_) / float(origin_w);
    }
  }

  int resize_h = int(float(origin_h) * ratio);
  int resize_w = int(float(origin_w) * ratio);
  resize_h = std::max(int(std::round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(std::round(float(resize_w) / 32) * 32), 32);
  int target_h = resize_h;
  int target_w = resize_w;
  if (desc.data_format_ == base::kDataFormatNCHW) {
    desc.shape_ = {
        1, preprocess::getChannelByPixelType(tmp_param->dst_pixel_type_),
        target_h, target_w};
  } else {
    desc.shape_ = {
        1, target_h, target_w,
        preprocess::getChannelByPixelType(tmp_param->dst_pixel_type_)};
  }
  device::Tensor *dst = outputs_[0]->create(device, desc);

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
    cv::resize(tmp_cvt, tmp_resize, cv::Size(resize_w, resize_h), 0.0, 0.0,
               interp_type);
  } else {
    tmp_resize = tmp_cvt;
  }
  tmp_param->top_ = 0;
  tmp_param->left_ = 0;
  tmp_param->bottom_ = target_h - resize_h;
  tmp_param->right_ = target_w - resize_w;

  cv::Mat tmp_pad;

  preprocess::OpenCvUtil::copyMakeBorder(
      tmp_resize, tmp_pad, tmp_param->top_, tmp_param->bottom_,
      tmp_param->left_, tmp_param->right_, tmp_param->border_type_,
      tmp_param->border_val_);

  preprocess::OpenCvConvert::convertToTensor(tmp_pad, dst,
                                             true,  // normalize
                                             tmp_param->scale_,
                                             tmp_param->mean_, tmp_param->std_);

  outputs_[0]->notifyWritten(dst);
  return base::kStatusCodeOk;
}

base::Status DetectorPostProcess::run() {
  DetectorPostParam *param = dynamic_cast<DetectorPostParam *>(param_.get());
  float det_db_thresh = param->det_db_thresh_;
  float det_db_box_thresh = param->det_db_box_thresh_;
  float det_db_unclip_ratio = param->det_db_unclip_ratio_;
  bool use_dilation = param->use_dilation_;

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getData();

  int batch = tensor->getShapeIndex(0);
  int channel = tensor->getShapeIndex(1);
  int height = tensor->getShapeIndex(2);
  int width = tensor->getShapeIndex(3);

  OCRResult *ocr_result = new OCRResult();
  int b = 0;
  float *data_batch = data;

  std::vector<float> pred(width * height, 0.0);
  std::vector<unsigned char> cbuf(width * height, ' ');

  for (int i = 0; i < width * height; i++) {
    pred[i] = float(data[i]);
    cbuf[i] = (unsigned char)((data[i]) * 255);
  }
  cv::Mat cbuf_map(height, width, CV_8UC1, (unsigned char *)cbuf.data());
  cv::Mat pred_map(height, width, CV_32F, (float *)pred.data());

  const double threshold = det_db_thresh * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (use_dilation) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }

  std::vector<std::vector<std::vector<int>>> boxes;
  boxes = util_post_processor_.BoxesFromBitmap(
      pred_map, bit_map, det_db_box_thresh, det_db_unclip_ratio,
      param->det_db_score_mode_);
  ocr_result->detector_resized_h = height;
  ocr_result->detector_resized_w = width;
  for (int i = 0; i < boxes.size(); i++) {
    std::array<int, 8> new_box;
    int k = 0;
    for (auto &vec : boxes[i]) {
      for (auto &e : vec) {
        new_box[k++] = e;
      }
    }
    ocr_result->boxes_.emplace_back(new_box);
  }

  outputs_[0]->set(ocr_result, false);
  return base::kStatusCodeOk;
}
REGISTER_NODE("nndeploy::ocr::DetectorPostProcess", DetectorPostProcess);
REGISTER_NODE("nndeploy::ocr::DetectorPreProcess", DetectorPreProcess);
REGISTER_NODE("nndeploy::ocr::DetectorGraph", DetectorGraph);

}  // namespace ocr
}  // namespace nndeploy
