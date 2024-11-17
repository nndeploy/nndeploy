#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/segment/result.h"
#include "nndeploy/segment/segment_anything/sam.h"

using namespace nndeploy;

class DrawMaskNode : public dag::Node {
 public:
  DrawMaskNode(const std::string &name,
               std::initializer_list<dag::Edge *> inputs,
               std::initializer_list<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {}
  virtual ~DrawMaskNode() {}

  virtual base::Status run() {
    cv::Mat *input_mat = inputs_[0]->getCvMat(this);
    segment::SegmentResult *result =
        (segment::SegmentResult *)inputs_[1]->getParam(this);
    device::Tensor *mask = result->mask_;
    cv::Mat mask_output(mask->getHeight(), mask->getWidth(), CV_32FC1,
                        mask->getData());
    cv::threshold(mask_output, mask_output, 0.0, 255.0, cv::THRESH_BINARY);
    mask_output.convertTo(mask_output, CV_8U);
    cv::Mat *output_mat = new cv::Mat(mask_output);
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
  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  // 检测模型的有向无环图graph名称
  // NNDEPLOY_SAM
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
  dag::Edge input("segment_in");
  // 有向无环图graph的输出边packert
  dag::Edge output("segment_out");

  // 创建检测模型有向无环图graph
  dag::Graph *segment_graph =
      dag::createGraph(name, inference_type, device_type, &input, &output,
                       model_type, is_path, model_value);
  if (segment_graph == nullptr) {
    NNDEPLOY_LOGE("segment_graph is nullptr");
    return -1;
  }

  graph->addNode(segment_graph);

  // 解码节点
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", &input);

  decode_node->setPath(input_path);
  graph->addNode(decode_node);

  // draw mask
  dag::Edge *draw_mask = graph->createEdge("draw_mask");
  dag::Node *draw_mask_node = graph->createNode<DrawMaskNode>(
      "DrawMaskNode", {&input, &output}, {draw_mask});

  // 编码节点
  codec::EncodeNode *encode_node = codec::createEncodeNode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", draw_mask);
  encode_node->setPath(ouput_path);
  graph->addNode(encode_node);

  base::Status status = graph->setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }
  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();

  graph->dump();
  segment_graph->dump();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  // 有向无环图Graphz运行
  NNDEPLOY_TIME_POINT_START("graph->run()");
  int size = decode_node->getSize();
  for (int i = 0; i < size; ++i) {
    graph->run();
  }

  NNDEPLOY_TIME_POINT_END("graph->run()");

  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph run failed");
    return -1;
  }

  // 有向无环图graphz反初始化
  NNDEPLOY_TIME_POINT_START("graph->deinit()");
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->deinit()");

  NNDEPLOY_TIME_PROFILER_PRINT("segment time profiler");

  // 有向无环图graphz销毁
  delete graph;
  delete segment_graph;

  NNDEPLOY_LOGE("hello world!\n");

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}
