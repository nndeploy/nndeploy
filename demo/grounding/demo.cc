/**
 * @file demo.cc
 * @brief YOLO-World / YOLOE-Prompt Open-Vocabulary Detection Demo
 *
 * Demonstrates dual-input (image + text features) open-vocabulary detection.
 *
 * Usage:
 *   # YOLO-World with pre-computed text features
 *   ./nndeploy_demo_grounding --name yolo_world \
 *     --input_path /path/to/image.jpg \
 *     --model_value /path/to/yolo_world.onnx \
 *     --text_feats_path /path/to/txt_feats.bin \
 *     --inference_type kInferenceTypeOnnxRuntime \
 *     --model_type kModelTypeOnnx \
 *     --num_classes 80 \
 *     --output_path /path/to/output.jpg
 *
 *   # YOLO-World with custom class names (CLIP text encoder required)
 *   ./nndeploy_demo_grounding --name yolo_world \
 *     --input_path /path/to/image.jpg \
 *     --model_value /path/to/yolo_world.onnx \
 *     --class_names "dog,cat,bird,person,car" \
 *     --clip_model_path /path/to/clip_text.onnx \
 *     --clip_tokenizer_path /path/to/tokenizer.json \
 *     --inference_type kInferenceTypeOnnxRuntime \
 *     --model_type kModelTypeOnnx \
 *     --output_path /path/to/output.jpg
 *
 *   # YOLOE-Prompt
 *   ./nndeploy_demo_grounding --name yoloe_prompt \
 *     --input_path /path/to/image.jpg \
 *     --model_value /path/to/yoloe_prompt.onnx \
 *     --text_feats_path /path/to/text_feats.bin \
 *     --inference_type kInferenceTypeOnnxRuntime \
 *     --model_type kModelTypeOnnx \
 *     --num_classes 80 \
 *     --text_dim 256 \
 *     --output_path /path/to/output.jpg
 *
 * Text Features:
 *   YOLO-World uses CLIP text features with dimension 512.
 *   YOLOE-Prompt uses YOLOE text encoder features (dimension varies by model).
 *   Features should be a float32 binary file with shape
 *   (1, num_classes, text_dim) in row-major order.
 *   If --text_feats_path is not provided, dummy features are generated.
 *
 * Custom Prompts (YOLO-World only):
 *   Instead of pre-computed text features, you can pass --class_names
 *   with a comma-separated list of class names. A CLIP text encoder
 *   ONNX model and a HuggingFace tokenizer.json are then used to
 *   encode the class names at runtime.
 *
 *   To convert CLIP BPE vocab to HF tokenizer.json, use:
 *     python tools/prepare_clip_tokenizer.py \
 *       --bpe_vocab /path/to/bpe_simple_vocab_16e6.txt \
 *       --output /path/to/tokenizer.json
 *
 * Output:
 *   Detection results are printed to console.
 *   If --output_path is provided, annotated image is saved.
 */

#include "flag.h"
#include <sstream>
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#ifdef ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLO_WORLD
#include "nndeploy/grounding/yolo_world/yolo_world.h"
#endif
#include "nndeploy/grounding/yoloe_prompt/yoloe_prompt.h"
#include "nndeploy/tokenizer/tokenizer.h"

using namespace nndeploy;

// 定义自定义命令行参数
DEFINE_string(text_feats_path, "",
              "Path to text features binary file (float32, row-major)");
DEFINE_int32(text_dim, 512, "Text feature dimension (YOLO-World: 512)");
DEFINE_string(class_names, "",
              "Comma-separated class names for custom prompts (e.g. 'dog,cat')");
DEFINE_string(clip_model_path, "", "Path to CLIP text encoder ONNX model");
DEFINE_string(clip_tokenizer_path, "",
              "Path to HuggingFace tokenizer.json for CLIP");
DEFINE_string(clip_inference_type,
              "kInferenceTypeOnnxRuntime",
              "Inference type for CLIP text encoder");
DEFINE_double(score_threshold, 0.5, "Score threshold for detection");
DEFINE_double(nms_threshold, 0.45, "NMS threshold for detection");
DEFINE_int32(num_classes, 80, "Number of detection classes");
DEFINE_int32(model_h, 640, "Model input height");
DEFINE_int32(model_w, 640, "Model input width");

/**
 * @brief 加载或生成文本特征张量
 *
 * 从二进制文件加载文本特征，若文件路径为空则生成虚拟特征。
 *
 * @param device 目标设备（CPU）
 * @param num_classes 类别数
 * @param text_dim 文本特征维度
 * @return device::Tensor* 文本特征张量
 */
device::Tensor* loadOrCreateTextFeatures(device::Device* device,
                                         int num_classes, int text_dim) {
  device::TensorDesc desc;
  desc.data_type_ = base::dataTypeOf<float>();
  desc.shape_ = {1, num_classes, text_dim};

  size_t feat_bytes = (size_t)num_classes * text_dim * sizeof(float);
  auto* buffer = new device::Buffer(device, feat_bytes);
  float* feat_data = static_cast<float*>(buffer->getData());

  if (!FLAGS_text_feats_path.empty()) {
    std::ifstream file(FLAGS_text_feats_path, std::ios::binary);
    if (!file.is_open()) {
      NNDEPLOY_LOGE("Failed to open text features file: %s\n",
                    FLAGS_text_feats_path.c_str());
      delete buffer;
      return nullptr;
    }
    file.read(reinterpret_cast<char*>(feat_data), feat_bytes);
    if (file.gcount() != (std::streamsize)feat_bytes) {
      NNDEPLOY_LOGW(
          "Text features file size mismatch: expected %zu bytes, got "
          "%zu bytes. Padding with zeros.\n",
          feat_bytes, (size_t)file.gcount());
    }
    NNDEPLOY_LOGI("Loaded text features from: %s (%zu bytes)\n",
                  FLAGS_text_feats_path.c_str(), (size_t)file.gcount());
  } else {
    for (int i = 0; i < num_classes; ++i) {
      for (int j = 0; j < text_dim; ++j) {
        if (i < text_dim && j == i) {
          feat_data[i * text_dim + j] = 1.0f;
        } else {
          feat_data[i * text_dim + j] = 0.01f;
        }
      }
    }
    NNDEPLOY_LOGW(
        "No --text_feats_path provided. Using DUMMY text features for "
        "testing.\n");
  }

  auto* tensor = new device::Tensor(desc, buffer, "text_feats");
  return tensor;
}

/**
 * @brief 在图像上绘制检测结果
 */
void drawDetections(cv::Mat& image, const detect::DetectResult& results) {
  for (const auto& bbox : results.bboxs_) {
    int x1 = static_cast<int>(bbox.bbox_[0] * image.cols);
    int y1 = static_cast<int>(bbox.bbox_[1] * image.rows);
    int x2 = static_cast<int>(bbox.bbox_[2] * image.cols);
    int y2 = static_cast<int>(bbox.bbox_[3] * image.rows);

    x1 = std::max(0, std::min(x1, image.cols - 1));
    y1 = std::max(0, std::min(y1, image.rows - 1));
    x2 = std::max(0, std::min(x2, image.cols - 1));
    y2 = std::max(0, std::min(y2, image.rows - 1));

    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(0, 255, 0), 2);

    char label[64];
    snprintf(label, sizeof(label), "cls:%d %.2f", bbox.label_id_, bbox.score_);
    cv::putText(image, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 255, 0), 1);
  }
}

std::vector<std::string> parseClassNames(const std::string& class_names_str) {
  std::vector<std::string> names;
  std::stringstream ss(class_names_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t\r\n"));
    token.erase(token.find_last_not_of(" \t\r\n") + 1);
    if (!token.empty()) {
      names.push_back(token);
    }
  }
  return names;
}

#ifdef ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLO_WORLD
/**
 * @brief 运行 YOLO-World 开放词汇检测
 *
 * Two modes:
 *   1. Pre-computed text features (--text_feats_path or dummy)
 *   2. Custom class names via CLIP (--class_names + --clip_model_path)
 */
int runYoloWorld(const std::string& model_path, const std::string& input_path,
                 const std::string& output_path,
                 base::InferenceType inference_type,
                 base::DeviceType device_type, base::ModelType model_type,
                 int num_classes, int text_dim, float score_threshold,
                 float nms_threshold, int model_h, int model_w,
                 const std::vector<std::string>& class_names,
                 const std::string& clip_model_path,
                 const std::string& clip_tokenizer_path) {
  grounding::YoloWorldGraph graph("yolo_world");

  graph.setInferenceType(inference_type);
  std::vector<std::string> model_values = {model_path};
  graph.setInferParam(device_type, model_type, true, model_values);
  graph.setScoreThreshold(score_threshold);
  graph.setNmsThreshold(nms_threshold);
  graph.setModelHW(model_h, model_w);

  bool use_custom_prompts = !class_names.empty();
#ifdef ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP
  if (use_custom_prompts) {
    graph.setNumClasses(static_cast<int>(class_names.size()));
    graph.setTextDim(text_dim);
    graph.setClassNames(class_names);
    graph.setClipModelPath(clip_model_path);
    graph.setClipTokenizerPath(clip_tokenizer_path);
  } else {
    graph.setNumClasses(num_classes);
    graph.setTextDim(text_dim);
  }
#else
  if (use_custom_prompts) {
    NNDEPLOY_LOGE(
        "CLIP text encoding requires tokenizer plugin. "
        "Build with ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP=ON.\n");
    return -1;
  }
  graph.setNumClasses(num_classes);
  graph.setTextDim(text_dim);
#endif

  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    NNDEPLOY_LOGE("Failed to load image: %s\n", input_path.c_str());
    return -1;
  }

  dag::Edge img_edge("image");
  dag::Edge txt_edge("txt_feats");
  img_edge.set(image);

  device::Tensor* text_feats = nullptr;
  if (!use_custom_prompts) {
    auto* device = device::getDevice(base::kDeviceTypeCodeCpu);
    text_feats = loadOrCreateTextFeatures(device, num_classes, text_dim);
    if (text_feats == nullptr) {
      return -1;
    }
    txt_edge.set(text_feats);
  } else {
#ifdef ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP
    tokenizer::TokenizerText clip_text;
    clip_text.texts_ = class_names;
    txt_edge.set((base::Param*)&clip_text);
#else
    NNDEPLOY_LOGE("CLIP text encoding requires tokenizer plugin.\n");
    return -1;
#endif
  }

  NNDEPLOY_LOGI("Running YOLO-World inference...\n");
  std::vector<dag::Edge*> outputs = graph({&img_edge, &txt_edge});

  if (outputs.empty()) {
    NNDEPLOY_LOGE("No output from graph\n");
    delete text_feats;
    return -1;
  }
  detect::DetectResult* result =
      outputs[0]->getGraphOutput<detect::DetectResult>();
  if (result == nullptr) {
    NNDEPLOY_LOGE("Failed to get DetectResult from output\n");
    delete text_feats;
    return -1;
  }

  NNDEPLOY_LOGI("Detection results: %zu objects found\n",
                result->bboxs_.size());
  for (size_t i = 0; i < result->bboxs_.size(); ++i) {
    const auto& bbox = result->bboxs_[i];
    NNDEPLOY_LOGI("  [%zu] label=%d score=%.4f bbox=[%.3f,%.3f,%.3f,%.3f]\n", i,
                  bbox.label_id_, bbox.score_, bbox.bbox_[0], bbox.bbox_[1],
                  bbox.bbox_[2], bbox.bbox_[3]);
  }

  if (!output_path.empty()) {
    drawDetections(image, *result);
    cv::imwrite(output_path, image);
    NNDEPLOY_LOGI("Annotated image saved to: %s\n", output_path.c_str());
  }

  delete text_feats;
  graph.deinit();

  return 0;
}
#endif  // ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLO_WORLD

/**
 * @brief 运行 YOLOE-Prompt 开放词汇检测
 */
int runYoloePrompt(const std::string& model_path, const std::string& input_path,
                   const std::string& output_path,
                   base::InferenceType inference_type,
                   base::DeviceType device_type, base::ModelType model_type,
                   int num_classes, int text_dim, float score_threshold,
                   float nms_threshold, int model_h, int model_w) {
  // 创建 YOLOE-Prompt 图
  grounding::YoloPromptGraph graph("yoloe_prompt");

  // 配置推理参数
  graph.setInferenceType(inference_type);
  std::vector<std::string> model_values = {model_path};
  graph.setInferParam(device_type, model_type, true, model_values);
  graph.setScoreThreshold(score_threshold);
  graph.setNmsThreshold(nms_threshold);
  graph.setModelHW(model_h, model_w);

  // 不调用 init()，直接使用 forward() 方法

  // 加载图像
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    NNDEPLOY_LOGE("Failed to load image: %s\n", input_path.c_str());
    return -1;
  }

  // 加载文本特征
  auto* device = device::getDevice(base::kDeviceTypeCodeCpu);
  device::Tensor* text_feats =
      loadOrCreateTextFeatures(device, num_classes, text_dim);
  if (text_feats == nullptr) {
    return -1;
  }

  // 创建输入边
  dag::Edge img_edge("image");
  dag::Edge txt_edge("text_feats");

  // 设置输入数据
  img_edge.set(image);
  txt_edge.set(text_feats);

  // 运行推理（触发自定义 forward）
  NNDEPLOY_LOGI("Running YOLOE-Prompt inference...\n");
  std::vector<dag::Edge*> outputs = graph({&img_edge, &txt_edge});

  // 读取检测结果
  if (outputs.empty()) {
    NNDEPLOY_LOGE("No output from graph\n");
    delete text_feats;
    return -1;
  }
  detect::DetectResult* result =
      outputs[0]->getGraphOutput<detect::DetectResult>();
  if (result == nullptr) {
    NNDEPLOY_LOGE("Failed to get DetectResult from output\n");
    delete text_feats;
    return -1;
  }

  // 打印结果
  NNDEPLOY_LOGI("Detection results: %zu objects found\n",
                result->bboxs_.size());
  for (size_t i = 0; i < result->bboxs_.size(); ++i) {
    const auto& bbox = result->bboxs_[i];
    NNDEPLOY_LOGI("  [%zu] label=%d score=%.4f bbox=[%.3f,%.3f,%.3f,%.3f]\n", i,
                  bbox.label_id_, bbox.score_, bbox.bbox_[0], bbox.bbox_[1],
                  bbox.bbox_[2], bbox.bbox_[3]);
  }

  // 保存标注图像
  if (!output_path.empty()) {
    drawDetections(image, *result);
    cv::imwrite(output_path, image);
    NNDEPLOY_LOGI("Annotated image saved to: %s\n", output_path.c_str());
  }

  // 清理
  delete text_feats;
  graph.deinit();

  return 0;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    std::cout << "  --text_feats_path: Path to text features binary file"
              << std::endl;
    std::cout << "  --text_dim: Text feature dimension (default: 512)"
              << std::endl;
    std::cout << "  --class_names: Comma-separated class names (e.g. 'dog,cat')"
              << std::endl;
    std::cout << "  --clip_model_path: Path to CLIP text encoder ONNX model"
              << std::endl;
    std::cout << "  --clip_tokenizer_path: Path to HuggingFace tokenizer.json"
              << std::endl;
    std::cout << "  --score_threshold: Score threshold (default: 0.5)"
              << std::endl;
    std::cout << "  --nms_threshold: NMS threshold (default: 0.45)"
              << std::endl;
    std::cout << "  --num_classes: Number of classes (default: 80)"
              << std::endl;
    std::cout << "  --model_h: Model input height (default: 640)" << std::endl;
    std::cout << "  --model_w: Model input width (default: 640)" << std::endl;
    return -1;
  }

  std::string name = demo::getName();
  std::string input_path = demo::getInputPath();
  std::string output_path = demo::getOutputPath();
  std::vector<std::string> model_values = demo::getModelValue();
  base::InferenceType inference_type = demo::getInferenceType();
  base::DeviceType device_type = demo::getDeviceType();
  base::ModelType model_type = demo::getModelType();

  int num_classes = FLAGS_num_classes;
  int text_dim = FLAGS_text_dim;
  float score_threshold = FLAGS_score_threshold;
  float nms_threshold = FLAGS_nms_threshold;
  int model_h = FLAGS_model_h;
  int model_w = FLAGS_model_w;
  std::vector<std::string> class_names;
  if (!FLAGS_class_names.empty()) {
    class_names = parseClassNames(FLAGS_class_names);
    num_classes = static_cast<int>(class_names.size());
  }

  if (input_path.empty()) {
    NNDEPLOY_LOGE("--input_path is required\n");
    return -1;
  }
  if (model_values.empty()) {
    NNDEPLOY_LOGE("--model_value is required\n");
    return -1;
  }
  if (name.empty()) {
    NNDEPLOY_LOGE("--name is required (yolo_world or yoloe_prompt)\n");
    return -1;
  }

  std::string model_path = model_values[0];

#ifdef ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLO_WORLD
  if (name == "yolo_world") {
    return runYoloWorld(model_path, input_path, output_path, inference_type,
                        device_type, model_type, num_classes, text_dim,
                        score_threshold, nms_threshold, model_h, model_w,
                        class_names, FLAGS_clip_model_path,
                        FLAGS_clip_tokenizer_path);
  } else
#endif
  if (name == "yoloe_prompt") {
    return runYoloePrompt(model_path, input_path, output_path, inference_type,
                          device_type, model_type, num_classes, text_dim,
                          score_threshold, nms_threshold, model_h, model_w);
  } else {
    NNDEPLOY_LOGE(
        "Unsupported algorithm: %s (use yolo_world or yoloe_prompt)\n",
        name.c_str());
    return -1;
  }
}
