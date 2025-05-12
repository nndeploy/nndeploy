
#include "nndeploy/preprocess/cvtcolor_resize.h"

#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

base::Status CvtColorResize::run() {
  // NNDEPLOY_LOGE("preprocess start!Thread ID: %d.\n",
  //               std::this_thread::get_id());
  CvtclorResizeParam *tmp_param =
      dynamic_cast<CvtclorResizeParam *>(param_.get());
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

  cv::Mat tmp_resize;
  if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
    int interp_type =
        OpenCvConvert::convertFromInterpType(tmp_param->interp_type_);
    cv::resize(tmp_cvt, tmp_resize, cv::Size(w, h), 0.0, 0.0, interp_type);
  } else {
    tmp_resize = tmp_cvt;
  }
  OpenCvConvert::convertToTensor(tmp_resize, dst, tmp_param->normalize_,
                                 tmp_param->scale_, tmp_param->mean_,
                                 tmp_param->std_);

  // 通知Edge，数据已经完成写入
  // dst->print();
  // dst->getDesc().print();
  outputs_[0]->notifyWritten(dst);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::preprocess::CvtColorResize", CvtColorResize);

}  // namespace preprocess
}  // namespace nndeploy
