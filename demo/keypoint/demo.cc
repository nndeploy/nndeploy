/**
 * @file demo.cc
 * @brief YOLOv8-Pose 姿态估计演示程序
 *
 * 基于 KeypointGraph 的 YOLOv8-Pose 关键点检测演示。
 *
 * 用法:
 *   ./nndeploy_demo_keypoint \
 *     --name nndeploy::keypoint::KeypointGraph \
 *     --inference_type kInferenceTypeOnnxRuntime \
 *     --device_type kDeviceTypeCodeX86:0 \
 *     --model_type kModelTypeOnnx \
 *     --is_path \
 *     --model_value /path/to/yolov8s-pose.onnx \
 *     --input_path /path/to/image.jpg \
 *     --output_path /path/to/output.jpg
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/keypoint/drawkeypoint.h"
#include "nndeploy/keypoint/result.h"
#include "nndeploy/keypoint/yolo_pose/yolo_pose.h"

using namespace nndeploy;
using namespace nndeploy::keypoint;

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);
  fprintf(stderr, "[KP-DEBUG] main() entered\n");
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    std::cout << "关键点检测演示程序 (YOLOv8-Pose)" << std::endl;
    std::cout << "  --name: 模型名称 (默认: nndeploy::keypoint::KeypointGraph)"
              << std::endl;
    std::cout << "  --input_path: 输入图像路径 (必填)" << std::endl;
    std::cout << "  --output_path: 输出图像路径 (可选)" << std::endl;
    std::cout << "  --model_value: 模型路径 (必填)" << std::endl;
    std::cout << "  --inference_type: 推理后端类型" << std::endl;
    std::cout << "  --device_type: 推理设备类型" << std::endl;
    std::cout << "  --model_type: 模型类型" << std::endl;
    return -1;
  }

  // 解析命令行参数
  fprintf(stderr, "[KP-DEBUG] Parsing flags...\n");
  std::string name = demo::getName();
  fprintf(stderr, "[KP-DEBUG] name='%s'\n", name.c_str());
  if (name.empty()) {
    name = "nndeploy::keypoint::KeypointGraph";
  }
  base::InferenceType inference_type = demo::getInferenceType();
  base::DeviceType device_type = demo::getDeviceType();
  base::ModelType model_type = demo::getModelType();
  bool is_path = demo::isPath();
  std::vector<std::string> model_value = demo::getModelValue();
  std::string input_path = demo::getInputPath();
  std::string output_path = demo::getOutputPath();
  fprintf(stderr,
          "[KP-DEBUG] Flags parsed: inference_type=%d device_type=%d "
          "model_type=%d\n",
          (int)inference_type, (int)device_type.code_, (int)model_type);

  // 参数校验
  if (input_path.empty()) {
    NNDEPLOY_LOGE("--input_path is required\n");
    return -1;
  }
  if (model_value.empty()) {
    NNDEPLOY_LOGE("--model_value is required\n");
    return -1;
  }

  // 创建 KeypointGraph
  fprintf(stderr, "[KP-DEBUG] Creating KeypointGraph...\n");
  KeypointGraph graph(name);
  fprintf(stderr, "[KP-DEBUG] KeypointGraph created\n");

  // 构建图：连接 preprocess -> infer -> postprocess 的边
  fprintf(stderr, "[KP-DEBUG] Calling make()...\n");
  dag::NodeDesc pre_desc("preprocess", {"keypoint_in"},
                         {"keypoint_preproc_out"});
  dag::NodeDesc infer_desc("infer", {"keypoint_preproc_out"},
                           {"keypoint_infer_out"});
  dag::NodeDesc post_desc("postprocess", {"keypoint_infer_out"},
                          {"keypoint_out"});
  base::Status status =
      graph.make(pre_desc, infer_desc, inference_type, post_desc);
  fprintf(stderr, "[KP-DEBUG] make() returned status=%d desc='%s'\n",
          (int)status.getStatusCode(), status.desc().c_str());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph make failed: %s\n", status.desc().c_str());
    return -1;
  }

  // 设置推理参数
  NNDEPLOY_LOGI("[DEBUG] Calling setInferParam...\n");
  graph.setInferParam(device_type, model_type, is_path, model_value);
  NNDEPLOY_LOGI("[DEBUG] setInferParam done\n");

  // 加载输入图像
  NNDEPLOY_LOGI("[DEBUG] Loading image: %s\n", input_path);
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    NNDEPLOY_LOGE("Failed to load image: %s\n", input_path.c_str());
    graph.deinit();
    return -1;
  }
  NNDEPLOY_LOGI("[DEBUG] Image loaded: %dx%d\n", image.cols, image.rows);

  // 创建输入边
  dag::Edge input_edge("keypoint_in");
  input_edge.set(image);
  NNDEPLOY_LOGI("[DEBUG] Input edge created\n");

  // 运行推理
  NNDEPLOY_LOGI("[DEBUG] Calling graph.forward()...\n");
  std::vector<dag::Edge*> outputs = graph.forward({&input_edge});
  NNDEPLOY_LOGI("[DEBUG] graph.forward() returned %zu outputs\n",
                outputs.size());

  if (outputs.empty()) {
    NNDEPLOY_LOGE("No output from graph\n");
    graph.deinit();
    return -1;
  }

  // 获取双输出结果：BBoxResult (edge 0) + KeypointResult (edge 1)
  detect::BBoxResult* bbox_result =
      outputs[0]->getGraphOutput<detect::BBoxResult>();
  KeypointResult* kp_result = outputs[1]->getGraphOutput<KeypointResult>();
  if (bbox_result == nullptr || kp_result == nullptr) {
    NNDEPLOY_LOGE("Failed to get results from graph output\n");
    graph.deinit();
    return -1;
  }

  NNDEPLOY_LOGI("Detection result: %zu person(s)\n",
                bbox_result->bboxs_.size());
  for (size_t d = 0; d < bbox_result->bboxs_.size(); ++d) {
    const auto& bbox = bbox_result->bboxs_[d];
    NNDEPLOY_LOGI("  person[%zu]: label_id=%d, score=%.4f\n", d, bbox.label_id_,
                  bbox.score_);
    NNDEPLOY_LOGI("    bbox=[%.3f, %.3f, %.3f, %.3f]\n", bbox.bbox_[0],
                  bbox.bbox_[1], bbox.bbox_[2], bbox.bbox_[3]);
  }

  NNDEPLOY_LOGI("Keypoint result: %zu skeleton(s)\n",
                kp_result->skeletons_.size());
  for (size_t d = 0; d < kp_result->skeletons_.size(); ++d) {
    const auto& sk = kp_result->skeletons_[d];
    NNDEPLOY_LOGI("  skeleton[%zu]: label_id=%d, score=%.4f\n", d, sk.label_id_,
                  sk.score_);
    NNDEPLOY_LOGI("    keypoints: %zu\n", sk.keypoints_.size());
    for (size_t i = 0; i < sk.keypoints_.size(); ++i) {
      const auto& kp = sk.keypoints_[i];
      NNDEPLOY_LOGI("      [%zu] (%.3f, %.3f) conf=%.3f\n", i, kp.x_, kp.y_,
                    kp.confidence_);
    }
  }

  if (!output_path.empty()) {
    cv::Mat drawn;
    image.copyTo(drawn);
    float img_w = static_cast<float>(drawn.cols);
    float img_h = static_cast<float>(drawn.rows);

    // 绘制边界框（来自 BBoxResult）
    for (const auto& bbox : bbox_result->bboxs_) {
      int x1 = static_cast<int>(bbox.bbox_[0] * img_w);
      int y1 = static_cast<int>(bbox.bbox_[1] * img_h);
      int x2 = static_cast<int>(bbox.bbox_[2] * img_w);
      int y2 = static_cast<int>(bbox.bbox_[3] * img_h);
      cv::rectangle(drawn, cv::Point(x1, y1), cv::Point(x2, y2),
                    cv::Scalar(0, 255, 0), 2);
      char label[64];
      snprintf(label, sizeof(label), "person %.2f", bbox.score_);
      cv::putText(drawn, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 255, 0), 1);
    }

    // 绘制关键点和骨架（来自 KeypointResult）
    for (const auto& sk : kp_result->skeletons_) {
      int num_kps = static_cast<int>(sk.keypoints_.size());
      for (int i = 0; i < 16; ++i) {
        int idx1 = kSkeleton[i][0];
        int idx2 = kSkeleton[i][1];
        if (idx1 >= num_kps || idx2 >= num_kps) continue;
        const KeypointKeyPoint& kp1 = sk.keypoints_[idx1];
        const KeypointKeyPoint& kp2 = sk.keypoints_[idx2];
        if (kp1.confidence_ < 0.5f || kp2.confidence_ < 0.5f) continue;
        cv::Point p1(static_cast<int>(kp1.x_ * img_w),
                     static_cast<int>(kp1.y_ * img_h));
        cv::Point p2(static_cast<int>(kp2.x_ * img_w),
                     static_cast<int>(kp2.y_ * img_h));
        cv::line(drawn, p1, p2, kLimbColors[kLimbGroup[i]], 2);
      }
      for (int i = 0; i < num_kps; ++i) {
        const KeypointKeyPoint& kp = sk.keypoints_[i];
        if (kp.confidence_ < 0.5f) continue;
        cv::Point pt(static_cast<int>(kp.x_ * img_w),
                     static_cast<int>(kp.y_ * img_h));
        cv::circle(drawn, pt, 4, kKpColor, -1);
        cv::circle(drawn, pt, 4, cv::Scalar(0, 0, 0), 1);
      }
    }

    cv::imwrite(output_path, drawn);
    NNDEPLOY_LOGI("Annotated image saved to: %s\n", output_path.c_str());
  }

  // 清理
  graph.deinit();

  return 0;
}
