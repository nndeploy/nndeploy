#include "nndeploy/preprocess/cvt_resize_crop_norm_trans.h"

#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

base::Status CvtResizeNormTransCropNormTrans::run() {
  CvtResizeNormTransCropNormTransParam *tmp_param =
      dynamic_cast<CvtResizeNormTransCropNormTransParam *>(param_.get());

  cv::Mat *src = inputs_[0]->getCvMat(this);

  device::Device *device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = tmp_param->data_type_;
  desc.data_format_ = tmp_param->data_format_;
  if (desc.data_format_ == base::kDataFormatNCHW) {
    desc.shape_ = {1, getChannelByPixelType(tmp_param->dst_pixel_type_),
                   tmp_param->height_, tmp_param->width_};
  } else {
    desc.shape_ = {1, tmp_param->height_, tmp_param->width_,
                   getChannelByPixelType(tmp_param->dst_pixel_type_)};
  }
  device::Tensor *dst = outputs_[0]->create(device, desc);

  int h = src->rows;
  int w = src->cols;

  cv::Mat tmp_cvt;
  if (tmp_param->src_pixel_type_ != tmp_param->dst_pixel_type_) {
    base::CvtColorType cvt_type = base::calCvtColorType(
        tmp_param->src_pixel_type_, tmp_param->dst_pixel_type_);
    if (cvt_type == base::kCvtColorTypeNotSupport) {
      NNDEPLOY_LOGE("cvtColor type not support");
      return base::kStatusCodeErrorNotSupport;
    }
    int cv_cvt_type = OpenCvConvert::convertFromCvtColorType(cvt_type);
    cv::cvtColor(*src, tmp_cvt, cv_cvt_type);
  } else {
    tmp_cvt = *src;
  }

  cv::Mat tmp_resize;
  int new_h, new_w;
  if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
    int interp_type =
        OpenCvConvert::convertFromInterpType(tmp_param->interp_type_);
    if (h < w) {
      float scale = static_cast<float>(tmp_param->resize_h_) / h;
      new_h = tmp_param->resize_h_;
      new_w = static_cast<int>(w * scale);
    } else {
      float scale = static_cast<float>(tmp_param->resize_w_) / w;
      new_w = tmp_param->resize_w_;
      new_h = static_cast<int>(h * scale);
    }
    cv::resize(tmp_cvt, tmp_resize, cv::Size(new_w, new_h), 0.0, 0.0,
               interp_type);
  } else {
    NNDEPLOY_LOGE("inerpolation type not support");
    return base::kStatusCodeErrorNotSupport;
  }

  cv::Mat tmp_crop;
  int start_x = (new_w - tmp_param->width_) / 2;
  int start_y = (new_h - tmp_param->height_) / 2;
  tmp_param->top_left_x_ =
      tmp_param->top_left_x_ ? tmp_param->top_left_x_ : start_x;
  tmp_param->top_left_y_ =
      tmp_param->top_left_y_ ? tmp_param->top_left_y_ : start_y;
  cv::Rect roi(tmp_param->top_left_x_, tmp_param->top_left_y_,
               tmp_param->width_, tmp_param->height_);
  tmp_crop = tmp_resize(roi).clone();

  OpenCvConvert::convertToTensor(tmp_crop, dst, tmp_param->normalize_,
                                 tmp_param->scale_, tmp_param->mean_,
                                 tmp_param->std_);
  outputs_[0]->notifyWritten(dst);

  // NNDEPLOY_LOGE("CvtResizeNormTransCropNormTrans run success\n");

  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::preprocess::CvtResizeNormTransCropNormTrans", CvtResizeNormTransCropNormTrans);

}  // namespace preprocess
}  // namespace nndeploy
