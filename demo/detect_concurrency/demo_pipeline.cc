#include <mutex>
#include <thread>

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/detect/yolo/yolo.h"

using namespace nndeploy;

cv::Mat drawBox(cv::Mat &cv_mat, model::DetectResult &result) {
  // float w_ratio = float(cv_mat.cols) / float(640);
  // float h_ratio = float(cv_mat.rows) / float(640);
  float w_ratio = float(cv_mat.cols);
  float h_ratio = float(cv_mat.rows);
  const int CNUM = 80;
  cv::RNG rng(0xFFFFFFFF);
  cv::Scalar_<int> randColor[CNUM];
  for (int i = 0; i < CNUM; i++)
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
  int i = -1;
  for (auto bbox : result.bboxs_) {
    std::array<float, 4> box;
    box[0] = bbox.bbox_[0];  // 640.0;
    box[2] = bbox.bbox_[2];  // 640.0;
    box[1] = bbox.bbox_[1];  // 640.0;
    box[3] = bbox.bbox_[3];  // 640.0;
    box[0] *= w_ratio;
    box[2] *= w_ratio;
    box[1] *= h_ratio;
    box[3] *= h_ratio;
    int width = box[2] - box[0];
    int height = box[3] - box[1];
    int id = bbox.label_id_;
    NNDEPLOY_LOGE("box[0]:%f, box[1]:%f, width :%d, height :%d\n", box[0],
                  box[1], width, height);
    cv::Point p = cv::Point(box[0], box[1]);
    cv::Rect rect = cv::Rect(box[0], box[1], width, height);
    cv::rectangle(cv_mat, rect, randColor[id]);
    std::string text = " ID:" + std::to_string(id);
    cv::putText(cv_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1, randColor[id]);
  }
  return cv_mat;
}

//
int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图graph名称，例如:
  // NNDEPLOY_YOLOV5/NNDEPLOY_YOLOV6/NNDEPLOY_YOLOV8
  std::string name = demo::getName();
  // 推理后端类型，例如:
  // kInferenceTypeOpenVino / kInferenceTypeTensorRt / kInferenceTypeOnnxRuntime
  base::InferenceType inference_type = demo::getInferenceType();
  // 推理设备类型，例如:
  // kDeviceTypeCodeX86:0/kDeviceTypeCodeCuda:0/...
  base::DeviceType device_type = demo::getDeviceType();
  // 模型类型，例如:
  // kModelTypeOnnx/kModelTypeMnn/...
  base::ModelType model_type = demo::getModelType();
  // 模型是否是路径
  bool is_path = demo::isPath();
  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();
  // 有向无环图graph的输入边packert
  dag::Edge input("detect_in");
  // 有向无环图graph的输出边packert
  dag::Edge output("detect_out");
  // 创建检测模型有向无环图graph
  dag::Graph *graph =
      dag::createGraph(name, inference_type, device_type, &input, &output,
                       model_type, is_path, model_value);
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  base::Status status = base::kStatusCodeOk;

  // 设置pipeline并行
  status = graph->setParallelType(dag::kParallelTypePipeline);

  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  NNDEPLOY_TIME_POINT_END("graph->init()");

  // 有向无环图graph的输入文件夹路径
  std::string input_path = demo::getInputPath();
  std::vector<std::string> all_file = demo::getAllFileFromDir(input_path);
  int index = 0;
  for (auto file_path : all_file) {
    NNDEPLOY_LOGI("file:%s\n", file_path.c_str());
    cv::Mat *input_mat = new cv::Mat(cv::imread(file_path));

    input.set(input_mat, index, false);

    graph->run();

    model::DetectResult *result =
        (model::DetectResult *)output.getParam(nullptr);
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr");
      return -1;
    }

    drawBox(*input_mat, *result);
    // std::string full_ouput_path = ouput_path + "/" + std::to_string(index) +
    //                               "_" + demo::getName() + ".jpg";

    // cv::imwrite(ouput_path, input_mat);

    index++;
  }

  std::string ouput_path = demo::getOutputPath();
  for (auto file_path : all_file) {
    NNDEPLOY_LOGI("file:%s\n", file_path.c_str());
    // cv::Mat input_mat = cv::imread(file_path);

    // model::DetectResult *result =
    //     (model::DetectResult *)output.getParam(nullptr);
    // if (result == nullptr) {
    //   NNDEPLOY_LOGE("result is nullptr");
    //   return -1;
    // }

    // drawBox(input_mat, *result);
    // std::string full_ouput_path = ouput_path + "/" + std::to_string(index)
    // +
    //                              "_" + demo::getName() + ".jpg";

    // cv::imwrite(ouput_path, input_mat);
  }

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }
  // 有向无环图graphz销毁
  delete graph;

  NNDEPLOY_LOGI("hello world!\n");

  return 0;
}
