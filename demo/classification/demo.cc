#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/classification/classification.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/codec/opencv/opencv_codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"
#include "nndeploy/classification/drawlabel.h"

using namespace nndeploy;

DEFINE_bool(is_softmax, false, "is_softmax");

bool isSoftmax() { return FLAGS_is_softmax; }


class classificationDemo : public dag::Graph {
 public:
  classificationDemo(const std::string &name) : dag::Graph(name) {}
  virtual ~classificationDemo() {}

  base::Status make(base::InferenceType inference_type,
                    base::CodecFlag codec_flag) {
    base::Status status = base::kStatusCodeOk;
    // 创建分类图
    decode_node_ = (codec::OpenCvImageDecode *)this
                       ->createNode<codec::OpenCvImageDecode>(
                           "decode_node_", codec_flag);
    graph_ =
        (classification::ClassificationGraph *)this
            ->createNode<classification::ClassificationGraph>("resnet");
    graph_->setInferenceType(inference_type);
    draw_node_ = (classification::DrawLable *)this->createNode<classification::DrawLable>(
        "draw_node", std::vector<dag::Edge *>(), std::vector<dag::Edge *>());
    encode_node_ = (codec::OpenCvImageEncode *)this
                       ->createNode<codec::OpenCvImageEncode>(
                           "encode_node_", codec_flag);
    return status;
  }

  base::Status setInferParam(base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string> &model_value) {
    graph_->setInferParam(device_type, model_type, is_path, model_value);
    return base::kStatusCodeOk;
  }

  base::Status setInputPath(const std::string &input_path) {
    decode_node_->setPath(input_path);
    return base::kStatusCodeOk;
  }

  base::Status setOutputPath(const std::string &output_path) {
    encode_node_->setPath(output_path);
    return base::kStatusCodeOk;
  }

  base::Status setRefPath(const std::string &ref_path) {
    encode_node_->setRefPath(ref_path);
    return base::kStatusCodeOk;
  }

  base::Status setSoftmax(bool is_softmax) {
    graph_->setSoftmax(is_softmax);
    return base::kStatusCodeOk;
  }

  virtual std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> decode_node_outputs = (*decode_node_)(inputs);

    std::vector<dag::Edge *> graph_outputs = (*graph_)(decode_node_outputs);

    std::vector<dag::Edge *> draw_node_inputs = {decode_node_outputs[0],
                                                 graph_outputs[0]};
    std::vector<dag::Edge *> draw_node_outputs =
        (*draw_node_)(draw_node_inputs);
    std::vector<dag::Edge *> encode_node_outputs =
        (*encode_node_)(draw_node_outputs);
    return graph_outputs;
  }

 public:
  codec::OpenCvImageDecode *decode_node_;
  codec::OpenCvImageEncode *encode_node_;
  classification::DrawLable *draw_node_;
  classification::ClassificationGraph *graph_;
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图graph名称，例如:nndeploy::classification::ClassificationGraph
  std::string name = demo::getName();
  // 推理后端类型，例如:
  // kInferenceTypeOpenVino / kInferenceTypeTensorRt /
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
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline / base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();
  // 后处理是否执行softmax
  bool is_softmax = isSoftmax();

  classificationDemo graph_demo("classification_demo");
  graph_demo.setTimeProfileFlag(true);
  graph_demo.make(inference_type, codec_flag);

  graph_demo.setInferParam(device_type, model_type, is_path, model_value);

  graph_demo.setSoftmax(is_softmax);

  // 设置pipeline并行
  base::Status status = graph_demo.setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }

  std::vector<dag::Edge *> inputs;
  // std::vector<dag::Edge *> outputs;

  std::vector<dag::Edge *> outputs = graph_demo.trace(inputs);

  graph_demo.setInputPath(input_path);
  graph_demo.setOutputPath(ouput_path);
  graph_demo.setRefPath(input_path);
  graph_demo.decode_node_->setSize(100);

  NNDEPLOY_TIME_POINT_START("graph_demo(inputs)");
  int size = 100;
  for (int i = 0; i < size; i++) {
    outputs = graph_demo(inputs);
    if (pt != base::kParallelTypePipeline) {
      classification::ClassificationResult *result =
          (classification::ClassificationResult *)outputs[0]
              ->getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  if (pt == base::kParallelTypePipeline) {
    for (int i = 0; i < size; ++i) {
      classification::ClassificationResult *result =
          (classification::ClassificationResult *)outputs[0]
              ->getGraphOutputParam();
      NNDEPLOY_LOGE("%d %p, %p.\n", i, result, outputs[0]);
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph_demo(inputs)");

  graph_demo.deinit();

  NNDEPLOY_TIME_PROFILER_PRINT("demo");
  NNDEPLOY_TIME_PROFILER_PRINT_REMOVE_WARMUP("demo", 10);

  return 0;
}