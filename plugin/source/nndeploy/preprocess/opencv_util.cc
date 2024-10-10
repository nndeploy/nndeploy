
#include "nndeploy/preprocess/opencv_util.h"

#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

void OpenCvUtil::cvtColor(cv::InputArray src, cv::OutputArray dst,
                          base::PixelType src_pixel_type,
                          base::PixelType dst_pixel_type) {
  if (src_pixel_type != dst_pixel_type) {
    base::CvtColorType cvt_type =
        base::calCvtColorType(src_pixel_type, dst_pixel_type);
    if (cvt_type == base::kCvtColorTypeNotSupport) {
      NNDEPLOY_LOGE("cvtColor type not support");
      return;
    }
    int cv_cvt_type = OpenCvConvert::convertFromCvtColorType(cvt_type);
    cv::cvtColor(src, dst, cv_cvt_type);
  }
}

void OpenCvUtil::cvtColor(cv::InputArray src, cv::OutputArray dst,
                          const CvtcolorParam &param) {
  OpenCvUtil::cvtColor(src, dst, param.src_pixel_type_, param.dst_pixel_type_);
}

void OpenCvUtil::resize(cv::InputArray src, cv::OutputArray dst, int dst_w,
                        int dst_h, float scale_w, float scale_h,
                        base::InterpType interp_type) {
  if (interp_type != base::kInterpTypeNotSupport) {
    int cv_interp_type = OpenCvConvert::convertFromInterpType(interp_type);
    cv::resize(src, dst, cv::Size(dst_w, dst_h), scale_w, scale_h,
               cv_interp_type);
  }
}

void OpenCvUtil::resize(cv::InputArray src, cv::OutputArray dst,
                        const ResizeParam &param) {
  OpenCvUtil::resize(src, dst, param.dst_w_, param.dst_h_, param.scale_w_,
                     param.scale_h_, param.interp_type_);
}

void OpenCvUtil::copyMakeBorder(cv::InputArray src, cv::OutputArray dst,
                                int top, int bottom, int left, int right,
                                base::BorderType border_type,
                                const base::Scalar2d &border_val) {
  int cv_border_type = OpenCvConvert::convertFromBorderType(border_type);
  cv::Scalar value = OpenCvConvert::convertFromScalar(border_val);
  cv::copyMakeBorder(src, dst, top, bottom, left, right, cv_border_type, value);
}

void OpenCvUtil::copyMakeBorder(cv::InputArray src, cv::OutputArray dst,
                                const PaddingParam &param) {
  OpenCvUtil::copyMakeBorder(src, dst, param.top_, param.bottom_, param.left_,
                             param.right_, param.border_type_,
                             param.border_val_);
}

void OpenCvUtil::warpAffine(cv::InputArray src, cv::OutputArray dst,
                            float *transform, int dst_w, int dst_h,
                            base::InterpType interp_type,
                            base::BorderType border_type,
                            const base::Scalar2d &border_val) {
  cv::Mat m;
  int cv_interp_type = OpenCvConvert::convertFromInterpType(interp_type);
  int cv_border_type = OpenCvConvert::convertFromBorderType(border_type);
  cv::Scalar value = OpenCvConvert::convertFromScalar(border_val);
  cv::warpAffine(src, dst, m, cv::Size(dst_w, dst_h), cv_interp_type,
                 cv_border_type, value);
}

void OpenCvUtil::warpAffine(cv::InputArray src, cv::OutputArray dst,
                            const WarpAffineParam &param) {
  OpenCvUtil::warpAffine(src, dst, (float *)(&param.transform_[0][0]),
                         param.dst_w_, param.dst_h_, param.interp_type_,
                         param.border_type_, param.border_val_);
}

cv::Mat OpenCvUtil::crop(cv::InputArray src, int x, int y, int w, int h) {
  cv::Rect rect(x, y, w, h);
  auto src_mat = src.getMat();
  cv::Mat dst = src_mat(rect);
  return dst;
}

cv::Mat OpenCvUtil::crop(cv::InputArray src, const CropParam &param) {
  cv::Rect rect(param.top_left_x_, param.top_left_y_, param.width_,
                param.height_);
  auto src_mat = src.getMat();
  cv::Mat dst = src_mat(rect);
  return dst;
}

}  // namespace preprocess
}  // namespace nndeploy