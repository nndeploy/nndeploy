#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/task/detect/opencv/detr.h"
#include "nndeploy/task/task.h"

using namespace nndeploy;

// cv::Mat draw_box(cv::Mat &cv_mat, task::DetectResult &result) {
//   float w_ratio = float(cv_mat.cols) / float(640);
//   float h_ratio = float(cv_mat.rows) / float(640);
//   int CNUM = 80;
//   cv::RNG rng(0xFFFFFFFF);
//   cv::Scalar_<int> randColor[CNUM];
//   for (int i = 0; i < CNUM; i++)
//     rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
//   int i = -1;
//   for (auto box : result.boxes_) {
//     ++i;
//     box[0] *= w_ratio;
//     box[2] *= w_ratio;
//     box[1] *= h_ratio;
//     box[3] *= h_ratio;
//     int width = box[2] - box[0];
//     int height = box[3] - box[1];
//     int id = result.label_ids_[i];
//     cv::Point p = cv::Point(box[0], box[1]);
//     cv::Rect rect = cv::Rect(box[0], box[1], width, height);
//     cv::rectangle(cv_mat, rect, randColor[result.label_ids_[i]]);
//     std::string text = " ID:" + std::to_string(id);
//     cv::putText(cv_mat, text, cv::Point(0, 0), cv::FONT_HERSHEY_PLAIN, 1,
//                 randColor[result.label_ids_[i]]);
//   }
//   return cv_mat;
// }

int main(int argc, char *argv[]) {
  base::TimeProfiler *tp = new base::TimeProfiler();

  std::string name = "opencv_detr";
  base::InferenceType type = base::kInferenceTypeMnn;
  bool is_path = true;
  std::vector<std::string> model_value;
  model_value.push_back(
      "/home/always/Downloads/TRT2021-MedcareAILab/detr_sim_onnx.mnn");

   task::Packet input;
   task::Packet output;
   task::Pipeline *pipeline = task::opencv::creatDetrPipeline(
       name, type, &input, &output, true, model_value);
   if (pipeline == nullptr) {
     NNDEPLOY_LOGE("pipeline is nullptr");
     return -1;
   }

   NNDEPLOY_LOGE("task->init()");

   pipeline->init();
   cv::Mat input_mat =
       cv::imread("/home/always/github/TensorRT-DETR/cpp/res.jpg");
   input.set(input_mat);

   NNDEPLOY_LOGE("task->run()");
   pipeline->run();

   // draw_box(input_mat, results);
   cv::imwrite("/home/always/github/TensorRT-DETR/cpp/res_again.jpg",
   input_mat);

   NNDEPLOY_LOGE("task->deinit()");
   pipeline->deinit();

   delete pipeline;

  printf("hello world!\n");
  return 0;
}
