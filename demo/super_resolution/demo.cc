#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/codec/opencv/batch_opencv_codec.h"
#include "nndeploy/codec/opencv/opencv_codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/super_resolution/super_resolution.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

void testCvVideoWriter(const std::string &input_path,
                       const std::string &ouput_path) {
  // // 读取输入视频
  auto cap = cv::VideoCapture(input_path);
  if (!cap.isOpened()) {
    NNDEPLOY_LOGE("Failed to open video file: %s\n", input_path.c_str());
    return;
  }
  int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
  NNDEPLOY_LOGI("frame_count: %d\n", frame_count);

  // 获取视频参数
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(cv::CAP_PROP_FPS);

  // 创建视频写入对象
  auto writer = cv::VideoWriter();
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  // int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  // std::string output_path = "vsr_src_output.mp4";
  // // if (ouput_path != output_path) {
  // //   NNDEPLOY_LOGI("输出路径不一致:\n");
  // //   NNDEPLOY_LOGI("  期望路径: %s\n", ouput_path.c_str());
  // //   NNDEPLOY_LOGI("  实际路径: %s\n", output_path.c_str());
  // // }
  // // output_path = ouput_path;
  // bool success = writer.open(output_path, fourcc, fps, cv::Size(width,
  // height), true); NNDEPLOY_LOGI("ouput_path:%s,\n", ouput_path.c_str());
  bool success =
      writer.open(ouput_path, fourcc, fps, cv::Size(width, height), true);
  if (!success) {
    NNDEPLOY_LOGE("Failed to create video writer for: %s\n",
                  ouput_path.c_str());
    cap.release();
    return;
  }
  NNDEPLOY_LOGI("writer is opened\n");

  // 逐帧读写
  cv::Mat frame;
  int index = 0;
  while (cap.read(frame)) {
    // cv::imshow("13",frame);
    // cv::waitKey(30);
    try {
      writer.write(frame);
    } catch (const cv::Exception &e) {
      NNDEPLOY_LOGE("Error writing frame: %s\n", e.what());
      break;
    }
    // std::string name = "output_temp" + std::to_string(index) + ".jpg";
    // std::string full_path = base::joinPath("./", name);
    // cv::imwrite(full_path, frame);
    index++;
  }

  // 释放资源
  writer.release();
  cap.release();
  NNDEPLOY_LOGI("writer is released\n");
}

class SuperResolutionDemo : public dag::Graph {
 public:
  SuperResolutionDemo(const std::string &name) : dag::Graph(name) {}
  virtual ~SuperResolutionDemo() {}

  base::Status make(base::InferenceType inference_type, int batch_size) {
    base::Status status = base::kStatusCodeOk;
    // 创建分类图
    decode_node_ =
        (codec::BatchOpenCvDecode *)this->createNode<codec::BatchOpenCvDecode>(
            "decode_node_");
    decode_node_->setNodeKey("nndeploy::codec::OpenCvVideoDecode");
    decode_node_->setBatchSize(batch_size);
    graph_ = (super_resolution::SuperResolutionGraph *)this
                 ->createNode<super_resolution::SuperResolutionGraph>("resnet");
    graph_->make(inference_type);
    encode_node_ =
        (codec::BatchOpenCvEncode *)this->createNode<codec::BatchOpenCvEncode>(
            "encode_node_");
    encode_node_->setNodeKey("nndeploy::codec::OpenCvVideoEncode");
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

  virtual std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> decode_node_outputs = (*decode_node_)(inputs);

    // std::vector<dag::Edge *> graph_outputs = (*graph_)(decode_node_outputs);

    // std::vector<dag::Edge *> encode_node_outputs =
    //     (*encode_node_)(graph_outputs);
    // return graph_outputs;

    std::vector<dag::Edge *> encode_node_outputs =
        (*encode_node_)(decode_node_outputs);
    return decode_node_outputs;
  }

 public:
  codec::BatchOpenCvDecode *decode_node_;
  codec::BatchOpenCvEncode *encode_node_;
  super_resolution::SuperResolutionGraph *graph_;
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图graph名称，例如:nndeploy::super_resolution::SuperResolutionGraph
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
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline / base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();

  SuperResolutionDemo graph_demo("super_resolution_demo");
  graph_demo.setTimeProfileFlag(true);
  int batch_size = 5;
  graph_demo.make(inference_type, batch_size);

  graph_demo.setInferParam(device_type, model_type, is_path, model_value);

  // 设置pipeline并行
  base::Status status = graph_demo.setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }

  std::vector<dag::Edge *> inputs;
  std::vector<dag::Edge *> outputs;

  // std::vector<dag::Edge *> outputs = graph_demo.trace(inputs);

  graph_demo.setInputPath(input_path);
  graph_demo.setRefPath(input_path);
  graph_demo.setOutputPath(ouput_path);

  NNDEPLOY_TIME_POINT_START("graph_demo(inputs)");
  int size = graph_demo.decode_node_->getSize();
  NNDEPLOY_LOGI("size: %d\n", size);
  for (int i = 0; i < size; i++) {
    outputs = graph_demo(inputs);
    if (pt != base::kParallelTypePipeline) {
      std::vector<cv::Mat> *result =
          outputs[0]->getGraphOutput<std::vector<cv::Mat>>();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  if (pt == base::kParallelTypePipeline) {
    for (int i = 0; i < size; ++i) {
      std::vector<cv::Mat> *result =
          outputs[0]->getGraphOutput<std::vector<cv::Mat>>();
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