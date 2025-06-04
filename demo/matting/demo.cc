#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/matting/pp_matting/pp_matting.h"

using namespace nndeploy;

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
  // base::kParallelTypePipeline or base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();
  std::vector<std::string> model_inputs = demo::getModelInputs();
  NNDEPLOY_LOGE("model_inputs = %s.\n", model_inputs[0].c_str());
  std::vector<std::string> model_outputs = demo::getModelOutputs();
  NNDEPLOY_LOGE("model_outputs = %s.\n", model_outputs[0].c_str());

  // 有向无环图graph的输入边packet
  dag::Edge input("matting_in");
  // 有向无环图graph的输出边packet
  dag::Edge output("matting_out");

  dag::Graph *graph = new dag::Graph("demo", {}, {&output});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }
  matting::PPMattingGraph *matting_graph =
      new matting::PPMattingGraph(name, {&input}, {&output});
  dag::NodeDesc pre_desc("preprocess", {"matting_in"}, model_inputs);
  dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
  dag::NodeDesc post_desc("postprocess", model_outputs, {"matting_out"});

  PPMattingGraph->make(pre_desc, infer_desc, inference_type, post_desc);
  PPMattingGraph->setInferParam(device_type, model_type, is_path, model_value);
  graph->addNode(segment_graph);

  // 解码节点
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", &input);
  graph->addNode(decode_node);

  return 0;
}
