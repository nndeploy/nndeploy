/**
 * nndeploy Segment Anything Demo:
 * Implementation of segment anything algorithm using static graph construction
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/codec/opencv/opencv_codec.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/segment/segment_anything/sam.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

DEFINE_string(point_label, "", "point label");

DEFINE_string(points, "", "points");

#define CHECK_IF_ERROR_RETURN(ret, fmt, ...) \
  if (ret != base::kStatusCodeOk) {          \
    NNDEPLOY_LOGE(fmt, ##__VA_ARGS__);       \
    return ret;                              \
  }

#define CHECK_IF_NULL_RETURN(ptr, fmt, ...)    \
  if (ptr == nullptr) {                        \
    NNDEPLOY_LOGE(fmt, ##__VA_ARGS__);         \
    return base::kStatusCodeErrorInvalidValue; \
  }

static std::vector<float> parsePoints(const std::string &points_str) {
  std::vector<float> points;
  std::istringstream iss(points_str);
  std::string point;
  while (std::getline(iss, point, ',')) {
    try {
      points.push_back(std::stof(point));
    } catch (const std::invalid_argument &) {
      NNDEPLOY_LOGE("Invalid point value: %s", point.c_str());
    }
  }
  return points;
}

class SAMDemo : public dag::Graph {
 public:
  SAMDemo(const std::string &name) : dag::Graph(name) {}

  virtual ~SAMDemo() {}

  base::Status makeGraph(base::CodecFlag codec_flag,
                         base::InferenceType inference_type,
                         base::DeviceType device_type,
                         base::ModelType model_type, bool is_path,
                         std::vector<std::string> &model_value) {
    base::Status status = base::kStatusCodeOk;

    dag::Edge *input_edge = this->createEdge("image_input");
    CHECK_IF_NULL_RETURN(input_edge, "Failed to create input edge");
    dag::Edge *output_edge = this->createEdge("SAM_result");
    CHECK_IF_NULL_RETURN(output_edge, "Failed to create output edge");
    dag::Edge *point_edge = this->createEdge("points_selected");
    CHECK_IF_NULL_RETURN(point_edge,
                         "Failed to create point edge for SAMGraph");

    std::vector<dag::Edge *> inputs, outputs;
    decode_node_ =
        (codec::OpenCvImageDecode *)this->createNode<codec::OpenCvImageDecode>(
            "decode_image", inputs, {input_edge}, codec_flag);
    CHECK_IF_NULL_RETURN(decode_node_,
                         "Failed to create OpenCvImageDecode node");

    encode_node_ =
        (codec::OpenCvImageEncode *)this->createNode<codec::OpenCvImageEncode>(
            "encode_image", {output_edge}, outputs, codec_flag);
    CHECK_IF_NULL_RETURN(encode_node_,
                         "Failed to create OpenCvImageEncode node");

    select_point_node_ =
        (segment::SelectPointNode *)this->createNode<segment::SelectPointNode>(
            "select_point_node", {input_edge}, {point_edge});

    sam_graph_ = (segment::SAMGraph *)this->createNode<segment::SAMGraph>(
        "sam_graph", {input_edge, point_edge}, {output_edge});
    CHECK_IF_NULL_RETURN(sam_graph_, "Failed to create SAMGraph node");

    status = sam_graph_->setInferParam(inference_type, device_type, model_type,
                                       is_path, model_value);
    CHECK_IF_ERROR_RETURN(status,
                          "Failed to set inference parameters for SAMGraph");

    return status;
  }

  base::Status makeDynamicsGraph(base::CodecFlag codec_flag,
                                 base::InferenceType inference_type,
                                 base::DeviceType device_type,
                                 base::ModelType model_type, bool is_path,
                                 std::vector<std::string> &model_value) {
    base::Status status = base::kStatusCodeOk;

    std::vector<dag::Edge *> inputs, outputs;
    decode_node_ =
        (codec::OpenCvImageDecode *)this->createNode<codec::OpenCvImageDecode>(
            "decode_image", codec_flag);
    CHECK_IF_NULL_RETURN(decode_node_,
                         "Failed to create OpenCvImageDecode node");

    encode_node_ =
        (codec::OpenCvImageEncode *)this->createNode<codec::OpenCvImageEncode>(
            "encode_image", codec_flag);
    CHECK_IF_NULL_RETURN(encode_node_,
                         "Failed to create OpenCvImageEncode node");

    select_point_node_ =
        (segment::SelectPointNode *)this->createNode<segment::SelectPointNode>(
            "select_point_node");

    sam_graph_ =
        (segment::SAMGraph *)this->createNode<segment::SAMGraph>("sam_graph");
    CHECK_IF_NULL_RETURN(sam_graph_, "Failed to create SAMGraph node");

    status = sam_graph_->setInferParam(inference_type, device_type, model_type,
                                       is_path, model_value);
    CHECK_IF_ERROR_RETURN(status,
                          "Failed to set inference parameters for SAMGraph");

    return status;
  }

  base::Status setPoints(const std::vector<float> &points,
                         const std::vector<float> &point_label) {
    CHECK_IF_NULL_RETURN(sam_graph_, "SAMGraph node is not initialized");
    CHECK_IF_NULL_RETURN(select_point_node_,
                         "SelectPointNode is not initialized");

    base::Status status = select_point_node_->setPoints(points, point_label);
    CHECK_IF_ERROR_RETURN(status, "Failed to set points in SelectPointNode")

    return status;
  }

  base::Status setInputPath(const std::string &input_path) {
    CHECK_IF_NULL_RETURN(decode_node_, "Decode node is not initialized");

    base::Status status = decode_node_->setPath(input_path);
    CHECK_IF_ERROR_RETURN(status, "Failed to set input path in Decode node");

    return status;
  }

  base::Status setOutputPath(const std::string &output_path) {
    CHECK_IF_NULL_RETURN(encode_node_, "Encode node is not initialized");

    base::Status status = encode_node_->setPath(output_path);
    CHECK_IF_ERROR_RETURN(status, "Failed to set output path in Encode node");

    return status;
  }

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> image =
        (*decode_node_)(inputs);  // Decode the input image
    std::vector<dag::Edge *> points =
        (*select_point_node_)(image);  // Select points

    std::vector<dag::Edge *> outputs =
        (*sam_graph_)({image[0], points[0]});  // Forward through SAMGraph

    std::vector<dag::Edge *> postprocess_output =
        (*encode_node_)(outputs);  // Encode the output

    return postprocess_output;  // Return the output edges
  }

 private:
  codec::OpenCvImageDecode *decode_node_ = nullptr;
  codec::OpenCvImageEncode *encode_node_ = nullptr;
  segment::SelectPointNode *select_point_node_ = nullptr;
  segment::SAMGraph *sam_graph_ = nullptr;
};

int main(int argc, char *argv[]) {
  NNDEPLOY_LOGI("Segment Anything Demo Start\n");

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }
  /**
   * @brief 获取图名称
   */
  std::string graph_name = demo::getName();
  /**
   * @brief 获取推理类型, kInferenceTypeOnnxRuntime
   */
  base::InferenceType inference_type = demo::getInferenceType();
  /**
   * @brief 获取设备类型, kDeviceTypeCodeX86:0
   */
  base::DeviceType device_type = demo::getDeviceType();
  /**
   * @brief 获取模型类型, kModelTypeOnnx
   */
  base::ModelType model_type = demo::getModelType();

  bool is_path = demo::isPath();
  std::vector<std::string> model_value = demo::getModelValue();

  // image path
  std::string input_path = demo::getInputPath();
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string output_path = demo::getOutputPath();

  base::ParallelType pt = demo::getParallelType();

  std::vector<float> point_label = parsePoints(FLAGS_point_label);

  std::vector<float> points = parsePoints(FLAGS_points);

  SAMDemo sam_demo(graph_name);
  base::Status status =
      sam_demo.makeDynamicsGraph(codec_flag, inference_type, device_type,
                                 model_type, is_path, model_value);
  CHECK_IF_ERROR_RETURN(status, "Failed to make SAMDemo graph");

  sam_demo.setTimeProfileFlag(true);

  status = sam_demo.setParallelType(pt);
  CHECK_IF_ERROR_RETURN(status, "Failed to set parallel type in SAMDemo");

  status = sam_demo.setPoints(points, point_label);
  CHECK_IF_ERROR_RETURN(status, "Failed to set points in SAMDemo");

  status = sam_demo.setInputPath(input_path);
  CHECK_IF_ERROR_RETURN(status, "Failed to set input path in SAMDemo");

  status = sam_demo.setOutputPath(output_path);
  CHECK_IF_ERROR_RETURN(status, "Failed to set output path in SAMDemo");

  // status = sam_demo.init();
  // CHECK_IF_ERROR_RETURN(status, "Failed to initialize SAMDemo graph");

  std::vector<dag::Edge *> inputs;
  // std::vector<dag::Edge *> outputs;

  std::vector<dag::Edge *> outputs = sam_demo.trace(inputs);

  status = sam_demo.dump();
  CHECK_IF_ERROR_RETURN(status, "Failed to dump SAMDemo graph");

  // sam_demo.trace();

  // status = sam_demo.run();
  outputs = sam_demo(inputs);
  CHECK_IF_ERROR_RETURN(status, "Failed to run SAMDemo graph");

  sam_demo.synchronize();

  sam_demo.deinit();

  NNDEPLOY_TIME_PROFILER_PRINT("demo");

  return 0;
}