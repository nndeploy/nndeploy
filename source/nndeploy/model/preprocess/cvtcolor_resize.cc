
#include "nndeploy/model/preprocess/cvtcolor_resize.h"

#include "nndeploy/model/preprocess/util.h"

namespace nndeploy {
namespace model {

base::Status CvtColorResize::run() {
  CvtclorResizeParam* tmp_param =
      dynamic_cast<CvtclorResizeParam*>(param_.get());
  cv::Mat* src = inputs_[0]->getCvMat(this);
  // device::Tensor* dst = outputs_[0]->getTensor();
  // if (dst->empty()) {
  //   device::TensorDesc desc = dst->getDesc();
  //   desc.data_type_ = base::dataTypeOf<float>();
  //   desc.data_format_ = base::kDataFormatNCHW;
  //   desc.shape_.emplace_back(1);
  //   desc.shape_.emplace_back(getChannelByPixelType(tmp_param->dst_pixel_type_));
  //   desc.shape_.emplace_back(tmp_param->h_);
  //   desc.shape_.emplace_back(tmp_param->w_);
  //   dst->justModify(desc);
  //   device::Device* device = device::getDefaultHostDevice();
  //   dst->allocBuffer(device);
  // }
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = base::dataTypeOf<float>();
  desc.data_format_ = base::kDataFormatNCHW;
  desc.shape_.emplace_back(1);
  desc.shape_.emplace_back(getChannelByPixelType(tmp_param->dst_pixel_type_));
  desc.shape_.emplace_back(tmp_param->h_);
  desc.shape_.emplace_back(tmp_param->w_);
  outputs_[0]->create(device, desc, inputs_[0]->getIndex(this));
  device::Tensor* dst = outputs_[0]->getTensor(this);

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

  cv::Mat tmp_resize;
  if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
    int interp_type =
        OpenCvConvert::convertFromInterpType(tmp_param->interp_type_);
    cv::resize(tmp_cvt, tmp_resize, cv::Size(w, h), 0.0, 0.0, interp_type);
  } else {
    tmp_resize = tmp_cvt;
  }

  OpenCvConvert::convertToTensor(tmp_resize, 56tk, tmp_param->scale_,
                                 tmp_param->mean_, tmp_param->std_);

  return base::kStatusCodeOk;
}

}  // namespace model
}  // namespace nndeploy
