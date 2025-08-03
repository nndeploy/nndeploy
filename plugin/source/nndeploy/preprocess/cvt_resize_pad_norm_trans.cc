#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"

#include "nndeploy/preprocess/opencv_util.h"
#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

base::Status CvtResizePadNormTrans::run() {
  CvtResizePadNormTransParam *tmp_param =
      dynamic_cast<CvtResizePadNormTransParam *>(param_.get());
  cv::Mat *src = inputs_[0]->getCvMat(this);
  device::Device *device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = tmp_param->data_type_;
  desc.data_format_ = tmp_param->data_format_;
  if (desc.data_format_ == base::kDataFormatNCHW) {
    desc.shape_ = {1, getChannelByPixelType(tmp_param->dst_pixel_type_),
                   tmp_param->h_, tmp_param->w_};
  } else {
    desc.shape_ = {1, tmp_param->h_, tmp_param->w_,
                   getChannelByPixelType(tmp_param->dst_pixel_type_)};
  }
  device::Tensor *dst = outputs_[0]->create(device, desc);

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
    int cv_cvt_type = OpenCvConvert::convertFromCvtColorType(cvt_type);
    cv::cvtColor(*src, tmp_cvt, cv_cvt_type);
  } else {
    tmp_cvt = *src;
  }

  int origin_h = src->rows;
  int origin_w = src->cols;
  float scale_h = (float)h / origin_h;
  float scale_w = (float)w / origin_w;
  int new_h, new_w;
  if (scale_h < scale_w) {
    new_w = std::round(origin_w * scale_h);
    new_h = h;
  } else {
    new_h = std::round(origin_h * scale_w);
    new_w = w;
  }
  cv::Mat tmp_resize;
  if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
    int interp_type =
        OpenCvConvert::convertFromInterpType(tmp_param->interp_type_);
    cv::resize(tmp_cvt, tmp_resize, cv::Size(new_w, new_h), 0.0, 0.0,
               interp_type);
  } else {
    tmp_resize = tmp_cvt;
  }

  tmp_param->top_ = 0;
  tmp_param->bottom_ = h - new_h - tmp_param->top_;
  tmp_param->left_ = 0;
  tmp_param->right_ = w - new_w - tmp_param->left_;
  cv::Mat tmp_pad;
  OpenCvUtil::copyMakeBorder(tmp_resize, tmp_pad, tmp_param->top_,
                             tmp_param->bottom_, tmp_param->left_,
                             tmp_param->right_, tmp_param->border_type_,
                             tmp_param->border_val_);

  OpenCvConvert::convertToTensor(tmp_pad, dst, tmp_param->normalize_,
                                 tmp_param->scale_, tmp_param->mean_,
                                 tmp_param->std_);

  // 通知Edge，数据已经完成写入
  outputs_[0]->notifyWritten(dst);

  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::preprocess::CvtResizePadNormTrans", CvtResizePadNormTrans);

}  // namespace preprocess
}  // namespace nndeploy
