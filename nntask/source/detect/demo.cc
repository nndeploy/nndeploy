#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nntask/source/detect/task.h"

cv::Mat draw_box(cv::Mat &cv_mat, nntask::common::DetectParam &result) {
  float w_ratio = float(cv_mat.cols) / float(640);
  float h_ratio = float(cv_mat.rows) / float(640);
  int CNUM = 80;
  cv::RNG rng(0xFFFFFFFF);
  cv::Scalar_<int> randColor[CNUM];
  for (int i = 0; i < CNUM; i++)
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
  int i = -1;
  for (auto box : result.boxes_) {
    ++i;
    box[0] *= w_ratio;
    box[2] *= w_ratio;
    box[1] *= h_ratio;
    box[3] *= h_ratio;
    int width = box[2] - box[0];
    int height = box[3] - box[1];
    int id = result.label_ids_[i];
    cv::Point p = cv::Point(box[0], box[1]);
    cv::Rect rect = cv::Rect(box[0], box[1], width, height);
    cv::rectangle(cv_mat, rect, randColor[result.label_ids_[i]]);
    std::string text = " ID:" + std::to_string(id);
    cv::putText(cv_mat, text, cv::Point(0, 0), cv::FONT_HERSHEY_PLAIN, 1,
                randColor[result.label_ids_[i]]);
  }
  return cv_mat;
}

int main(int argc, char *argv[]) {
  nndeploy::base::TimeMeasurement *tm = new nndeploy::base::TimeMeasurement();

  nndeploy::base::DeviceType device_type(nndeploy::base::kDeviceTypeCodeX86, 0);
  nntask::detect::Task *task = new nntask::detect::Task(
      true, nndeploy::base::kInferenceTypeMnn, device_type, "yolo");

  nndeploy::inference::InferenceParam *inference_param =
      (nndeploy::inference::InferenceParam *)(task->getInferenceParam());
  inference_param->is_path_ = true;
  inference_param->model_value_.push_back("/home/always/Downloads/yolov5s.mnn");

  task->init();

  cv::Mat input_mat = cv::imread("/home/always/Downloads/yolo_input.jpeg");
  nntask::common::Packet input(input_mat);
  nntask::common::DetectParam result;
  nntask::common::Packet output(result);

  task->setInput(input);
  task->setOutput(output);
  task->run();

  draw_box(input_mat, result);

  cv::imwrite("/home/always/Downloads/yolo_result.jpg", input_mat);

  task->deinit();

  printf("hello world!\n");
  return 0;
}