#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"
#include "nndeploy/source/task/detect/opencv/detr.h"
#include "nndeploy/source/task/detect/opencv/yolo.h"
#include "nndeploy/source/task/task.h"

using namespace nndeploy;

cv::Mat draw_box(cv::Mat &cv_mat, task::DetectResult &result) {
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
  base::TimeMeasurement *tm = new base::TimeMeasurement();

  std::string name = "opencv_yolov5";
  base::InferenceType type = base::kInferenceTypeMnn;

  task::Task *task = task::creteTask(name, type);
  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(task->getInferenceParam());
  inference_param->is_path_ = true;
  inference_param->model_value_.push_back("/home/always/Downloads/yolov5s.mnn");
  inference_param->device_type_ = device::getDefaultHostDeviceType();

  task->init();

  cv::Mat input_mat = cv::imread("/home/always/Downloads/yolo_input.jpeg");
  task::Packet input(input_mat);
  task::DetectResult result;
  task::Packet output(result);
  task->setInput(input);
  task->setOutput(output);
  task->run();

  draw_box(input_mat, result);
  cv::imwrite("/home/always/Downloads/yolo_result.jpg", input_mat);
  task->deinit();

  printf("hello world!\n");
  return 0;
}
