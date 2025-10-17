#include "nndeploy/ocr/classifier.h"

#include "nndeploy/ocr/result.h"
#include "nndeploy/preprocess/opencv_convert.h"
#include "nndeploy/preprocess/opencv_util.h"
#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace ocr {

bool SingleBatchPostprocessor(const float* out_data, const size_t& length,
                              int* cls_label, float* cls_score) {
  *cls_label = std::distance(&out_data[0],
                             std::max_element(&out_data[0], &out_data[length]));

  *cls_score = float(*std::max_element(&out_data[0], &out_data[length]));
  return true;
}

base::Status ClassifierPostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("version_", version_, allocator);
  json.AddMember("cls_thresh_", cls_thresh_, allocator);
  return base::kStatusCodeOk;
}

base::Status ClassifierPostParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }

  if (json.HasMember("cls_thresh_") && json["cls_thresh_"].IsFloat()) {
    cls_thresh_ = json["cls_thresh_"].GetFloat();
  }

  return base::kStatusCodeOk;
}

base::Status ClassifierPreProcess::run() {
  ClassifierPreProcessParam* tmp_param =
      dynamic_cast<ClassifierPreProcessParam*>(param_.get());
  auto edge_ptr = inputs_[0]->get<OCRResult>(this);
  auto detector_result = static_cast<OCRResult*>(edge_ptr);
  std::vector<int32_t>* cls_labels_ptr = &detector_result->cls_labels_;
  std::vector<float>* cls_scores_ptr = &detector_result->cls_scores_;

  std::vector<std::string>* text_ptr = &detector_result->text_;
  std::vector<float>* rec_scores_ptr = &detector_result->rec_scores_;
  int batch_size = detector_result->image_list_.size();
  int img_c = tmp_param->cls_image_shape_[0];
  int img_h = tmp_param->cls_image_shape_[1];
  int img_w = tmp_param->cls_image_shape_[2];
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = tmp_param->data_type_;
  desc.data_format_ = tmp_param->data_format_;
  desc.shape_ = {batch_size, img_c, img_h, img_w};
  device::Tensor* dst = outputs_[0]->create(device, desc);
  for (int b = 0; b < batch_size; b++) {
    cv::Mat img = detector_result->image_list_[b];
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
      cv::cvtColor(img, tmp_cvt, cv_cvt_type);
    } else {
      tmp_cvt = img;
    }

    float ratio = float(img.cols) / float(img.rows);

    int resize_w;
    if (ceilf(img_h * ratio) > img_w)
      resize_w = img_w;
    else
      resize_w = int(ceilf(img_h * ratio));
    cv::Mat tmp_resize;
    if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
      int interp_type = preprocess::OpenCvConvert::convertFromInterpType(
          tmp_param->interp_type_);
      cv::resize(tmp_cvt, tmp_resize, cv::Size(resize_w, img_h), 0.0, 0.0,
                 interp_type);
    } else {
      tmp_resize = tmp_cvt;
    }
    cv::Mat tmp_pad;
    if (img.cols < tmp_param->cls_image_shape_[2]) {
      tmp_param->top_ = 0;
      tmp_param->left_ = 0;
      tmp_param->bottom_ = 0;
      tmp_param->right_ = tmp_param->cls_image_shape_[2] - img.cols;
      preprocess::OpenCvUtil::copyMakeBorder(
          tmp_resize, tmp_pad, tmp_param->top_, tmp_param->bottom_,
          tmp_param->left_, tmp_param->right_, tmp_param->border_type_,
          tmp_param->border_val_);
    } else {
      tmp_pad = tmp_resize;
    }

    preprocess::OpenCvConvert::convertToBatchTensor(
        tmp_pad, dst,
        true,  // normalize
        tmp_param->scale_, tmp_param->mean_, tmp_param->std_, b);
  }

  int n = dst->getBatch();
  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();
  outputs_[0]->notifyWritten(dst);

  return base::kStatusCodeOk;
}

base::Status ClassifierPostProcess::run() {
  ClassifierPostParam* param = (ClassifierPostParam*)param_.get();
  float cls_thresh_ = param->cls_thresh_;

  device::Tensor* tensor = inputs_[0]->getTensor(this);
  float* data = (float*)tensor->getData();

  int batch = tensor->getShapeIndex(0);
  int channel = tensor->getShapeIndex(1);
  int height = tensor->getShapeIndex(2);
  int width = tensor->getShapeIndex(3);

  int h = tensor->getShapeIndex(2) > 0 ? tensor->getShapeIndex(2) : 1;
  int w = tensor->getShapeIndex(3) > 0 ? tensor->getShapeIndex(3) : 1;

  size_t length = channel * h * w;
  const float* out_data = (const float*)tensor->getData();

  std::vector<int> cls_labels(batch);
  std::vector<float> cls_scores(batch);

  OCRResult* ocr_result = new OCRResult();

  for (int i = 0; i < batch; i++) {
    const float* data_ptr = out_data + i * length;
    SingleBatchPostprocessor(data_ptr, length, &cls_labels[i], &cls_scores[i]);
    if (cls_labels[i] % 2 == 1 && cls_scores[i] > param->cls_thresh_) {
      ocr_result->classifier_result.push_back(1);
    } else {
      ocr_result->classifier_result.push_back(0);
    }
  }

  outputs_[0]->set(ocr_result, false);
  return base::kStatusCodeOk;
}
REGISTER_NODE("nndeploy::ocr::ClassifierPostProcess", ClassifierPostProcess);
REGISTER_NODE("nndeploy::ocr::ClassifierPreProcess", ClassifierPreProcess);
REGISTER_NODE("nndeploy::ocr::ClassifierGraph", ClassifierGraph);

}  // namespace ocr
}  // namespace nndeploy
