#include "nndeploy/preprocess/warpaffine_preprocess.h"

#include "nndeploy/preprocess/opencv_util.h"
#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

base::Status WarpaffinePreprocess::run() {
  WarpAffineParam* tmp_param = dynamic_cast<WarpAffineParam*>(param_.get());
  cv::Mat* src = inputs_[0]->getCvMat(this);

  unsigned char* input_mat = (*src).data;

  device::Device* device = device::getDefaultHostDevice();
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
  device::Tensor* dst = outputs_[0]->create(device, desc);

  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();

  int origin_h = src->rows;
  int origin_w = src->cols;
  float scale_h = h / (float)origin_h;
  float scale_w = w / (float)origin_w;
  float scale = std::min(scale_h, scale_w);

  float i2d[6];
  float d2i[6];

  i2d[0] = scale;
  i2d[1] = 0;
  i2d[2] = (-scale * origin_w + w + scale - 1) * 0.5;
  i2d[3] = 0;
  i2d[4] = scale;
  i2d[5] = (-scale * origin_h + h + scale - 1) * 0.5;

  cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
  cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
  cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

  cv::Mat input_image(h, w, CV_8UC3);
  cv::warpAffine(*src, input_image, m2x3_i2d, input_image.size(),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));

  int image_area = input_image.cols * input_image.rows;
  unsigned char* pimage = input_image.data;

  float* p_input = (float*)dst->getData();
  if (desc.data_format_ == base::kDataFormatNHWC) {
    for (int i = 0; i < image_area; ++i, pimage += 3) {
      *p_input++ = (pimage[2] - tmp_param->mean_[2]) / tmp_param->std_[3];
      *p_input++ = (pimage[1] - tmp_param->mean_[1]) / tmp_param->std_[2];
      *p_input++ = (pimage[0] - tmp_param->mean_[0]) / tmp_param->std_[1];
    }
  } else if (desc.data_format_ == base::kDataFormatNCHW) {
    float* phost_b = p_input + image_area * 0;
    float* phost_g = p_input + image_area * 1;
    float* phost_r = p_input + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3) {
      // 注意这里的顺序rgb调换了
      *phost_r++ = (pimage[0] - tmp_param->mean_[0]) / tmp_param->std_[1];
      *phost_g++ = (pimage[1] - tmp_param->mean_[1]) / tmp_param->std_[2];
      *phost_b++ = (pimage[2] - tmp_param->mean_[2]) / tmp_param->std_[3];
    }
  }

  // 通知Edge，数据已经完成写入
  outputs_[0]->notifyWritten(dst);

  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::preprocess::WarpaffinePreprocess",
              WarpaffinePreprocess);

}  // namespace preprocess
}  // namespace nndeploy
