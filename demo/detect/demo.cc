#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/detect/yolo/yolo.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

class DrawBoxNode : public dag::Node {
 public:
  DrawBoxNode(const std::string &name,
              std::initializer_list<dag::Edge *> inputs,
              std::initializer_list<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {}
  virtual ~DrawBoxNode() {}

  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    model::DetectResult *result =
        (model::DetectResult *)inputs_[1]->getParam(this);
    float w_ratio = float(input_mat->cols);
    float h_ratio = float(input_mat->rows);
    const int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
      rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    int i = -1;
    for (auto bbox : result->bboxs_) {
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
      // NNDEPLOY_LOGE("box[0]:%f, box[1]:%f, width :%d, height :%d\n", box[0],
      //               box[1], width, height);
      cv::Point p = cv::Point(box[0], box[1]);
      cv::Rect rect = cv::Rect(box[0], box[1], width, height);
      cv::rectangle(*input_mat, rect, randColor[id]);
      std::string text = " ID:" + std::to_string(id);
      cv::putText(*input_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1,
                  randColor[id]);
    }
    cv::Mat *output_mat = new cv::Mat(*input_mat);
    outputs_[0]->set(output_mat, inputs_[0]->getIndex(this), false);
    return base::kStatusCodeOk;
  }
};

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
  // input path
  std::string input_path = demo::getInputPath();
  // input path
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline
  // base::ParallelType pt = base::kParallelTypePipeline;
  base::ParallelType pt = base::kParallelTypeSequential;

  // graph
  dag::Graph *graph = new dag::Graph("demo", nullptr, nullptr);
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  // 有向无环图graph的输入边packert
  dag::Edge input("detect_in");
  // 有向无环图graph的输出边packert
  dag::Edge output("detect_out");
  // 创建检测模型有向无环图graph
  dag::Graph *detect_graph =
      dag::createGraph(name, inference_type, device_type, &input, &output,
                       model_type, is_path, model_value);
  if (detect_graph == nullptr) {
    NNDEPLOY_LOGE("detect_graph is nullptr");
    return -1;
  }
  graph->addNode(detect_graph);

  // 解码节点
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", &input);
  decode_node->setPath(input_path);
  graph->addNode(decode_node);

  // draw box
  dag::Edge *draw_output = graph->createEdge("draw_output");
  dag::Node *draw_box_node = graph->createNode<DrawBoxNode>(
      "DrawBoxNode", {&input, &output}, {draw_output});

  // 解码节点
  codec::EncodeNode *encode_node = codec::createEncodeNode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", draw_output);
  encode_node->setPath(ouput_path);
  graph->addNode(encode_node);

  // 设置pipeline并行
  base::Status status = graph->setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }

  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  status = graph->dump();
  status = detect_graph->dump();

  int size = decode_node->getSize();
  for (int i = 0; i < size; ++i) {
    graph->run();
  }

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  // 有向无环图graph销毁
  delete detect_graph;
  delete graph;

  NNDEPLOY_LOGI("hello world!\n");

  return 0;
}
