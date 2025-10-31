#include "nndeploy/segment/drawmask.h"

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace segment {

base::Status DrawMask::run() {
  // �ӵ�һ�������Ե��ȡ����ͼ�����
  cv::Mat *input_mat = inputs_[0]->getCvMat(this);
  // �ӵڶ��������Ե��ȡ�ָ���
  segment::SegmentResult *result =
      (segment::SegmentResult *)inputs_[1]->getParam(this);
  // ��ȡ��������
  device::Tensor *mask = result->mask_;
  // ���������������Ϊ������
  if (mask->getDataType() == base::dataTypeOf<float>()) {
    // ����һ�������������ߴ���ͬ�ĵ�ͨ���������
    cv::Mat mask_output(mask->getHeight(), mask->getWidth(), CV_32FC1,
                        mask->getData());
    // ����������ֵ��
    cv::threshold(mask_output, mask_output, 0.0, 255.0, cv::THRESH_BINARY);
    // ���������ת��Ϊ8λ�޷�����������
    mask_output.convertTo(mask_output, CV_8U);
    // �������ͼ�����
    cv::Mat *output_mat = new cv::Mat(mask_output);
    // ���������Ե������
    outputs_[0]->set(output_mat, false);
  } else if (mask->getDataType() == base::dataTypeOf<uint8_t>()) {
    // ���������������ߴ���ͬ��8λ�޷�����������
    cv::Mat mask_mat(mask->getHeight(), mask->getWidth(), CV_8UC1,
                     mask->getData());
    cv::Mat mask_result;
    // �����������ĳߴ���ƥ������ͼ��ĳߴ�
    cv::resize(mask_mat, mask_result, input_mat->size(), 0.0, 0.0,
               cv::INTER_LINEAR);
    // �������ͼ�����������ͼ�������ͬ��ͨ����
    cv::Mat *output_mat = nullptr;
    int channels = input_mat->channels();
    if (channels == 1) {
      output_mat = new cv::Mat(input_mat->size(), CV_8UC1, cv::Scalar(0));
    } else if (channels == 3) {
      output_mat = new cv::Mat(input_mat->size(), CV_8UC3, cv::Scalar(0, 0, 0));
    } else if (channels == 4) {
      output_mat =
          new cv::Mat(input_mat->size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    }

    // ����ÿ�����أ����������50�����ظ��Ƶ����ͼ����
    for (int y = 0; y < input_mat->rows; ++y) {
      for (int x = 0; x < input_mat->cols; ++x) {
        if (mask_result.at<uchar>(y, x) > 50) {
          if (channels == 1) {
            output_mat->at<uchar>(y, x) = input_mat->at<uchar>(y, x);
          } else if (channels == 3) {
            output_mat->at<cv::Vec3b>(y, x) = input_mat->at<cv::Vec3b>(y, x);
          } else if (channels == 4) {
            output_mat->at<cv::Vec4b>(y, x) = input_mat->at<cv::Vec4b>(y, x);
          }
        }
      }
    }

    // ���������Ե������
    outputs_[0]->set(output_mat, false);
  }
  // ���ز����ɹ�״̬
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::segment::DrawMask", DrawMask);

}  // namespace segment
}  // namespace nndeploy
