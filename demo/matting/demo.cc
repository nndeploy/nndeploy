/**
 * nndeploy Matting Demo:
 * Implementation of matting algorithm using static graph construction
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/matting/pp_matting/pp_matting.h"
#include "nndeploy/matting/vis_matting.h"

using namespace nndeploy;

DEFINE_int32(matting_model_size, 512, "matting_model_size");

int getMattingModelSize() { return FLAGS_matting_model_size; }

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图graph名称
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
  std::string output_path = demo::getOutputPath();
  // input path
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline or base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();
  std::vector<std::string> model_inputs = demo::getModelInputs();
  NNDEPLOY_LOGE("model_inputs = %s.\n", model_inputs[0].c_str());
  std::vector<std::string> model_outputs = demo::getModelOutputs();
  NNDEPLOY_LOGE("model_outputs = %s.\n", model_outputs[0].c_str());

  dag::Edge *input = new dag::Edge("matting_in");
  dag::Edge *output = new dag::Edge("matting_out");

  dag::Graph *graph = new dag::Graph("demo", {}, {output});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }
  matting::PPMattingGraph *matting_graph =
      new matting::PPMattingGraph(name, {input}, {output});
  dag::NodeDesc pre_desc("preprocess", {"matting_in"}, model_inputs);
  dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
  std::vector<std::string> post_inputs;
  post_inputs.push_back("matting_in");
  for (const auto &output : model_outputs) {
    post_inputs.push_back(output);
  }
  dag::NodeDesc post_desc("postprocess", post_inputs, {"matting_out"});

  matting_graph->make(pre_desc, infer_desc, inference_type, post_desc);
  matting_graph->setInferParam(device_type, model_type, is_path, model_value);
  int height = getMattingModelSize();
  int width = getMattingModelSize();
  matting_graph->setModelHW(height, width);
  graph->addNode(matting_graph);

  // 解码节点
  codec::Decode *decode_node = codec::createDecode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", input);
  graph->addNode(decode_node);

  dag::Edge *vis_matting_img = graph->createEdge("vis_matting_img");
  dag::Node *vis_matting_node;
  vis_matting_node = graph->createNode<matting::VisMatting>(
      "vis_matting_node", {input, output}, {vis_matting_img});

  codec::Encode *encode_node = codec::createEncode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", vis_matting_img);
  graph->addNode(encode_node);

  // Set pipeline
  base::Status status = graph->setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }

  graph->setTimeProfileFlag(true);

  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  status = graph->dump();

  NNDEPLOY_TIME_POINT_START("graph->run");
  decode_node->setPath(input_path);
  encode_node->setRefPath(input_path);
  encode_node->setPath(output_path);
  int size = 100;
  decode_node->setSize(size);
  for (int i = 0; i < size; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    if (pt != base::kParallelTypePipeline) {
      matting::MattingResult *result =
          (matting::MattingResult *)output->getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }

  if (pt == base::kParallelTypePipeline) {
    NNDEPLOY_LOGE("size = %d.\n", size);
    for (int i = 0; i < size; ++i) {
      matting::MattingResult *result =
          (matting::MattingResult *)output->getGraphOutputParam();
      NNDEPLOY_LOGE("%d %p.\n", i, result);
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  NNDEPLOY_TIME_PROFILER_PRINT("demo");

  delete input;
  delete output;
  delete encode_node;
  delete decode_node;
  delete matting_graph;
  delete graph;

  return 0;
}
