
# 构图方式的讨论

## 结论
- 建议使用继承，而非createGraph方法
- 还需支持动态构图

## 代码
```cpp
#include "nndeploy/classification/classification.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/classification/util.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_softmax.h"
#include "nndeploy/preprocess/cvtcolor_resize.h"

namespace nndeploy {
namespace classification {

base::Status ClassificationPostProcess::run() {
  ClassificationPostParam *param = (ClassificationPostParam *)param_.get();

  device::Tensor *tensor = inputs_[0]->getTensor(this);

  // tensor->print();
  // tensor->getDesc().print();

  std::shared_ptr<ir::SoftmaxParam> op_param =
      std::make_shared<ir::SoftmaxParam>();
  op_param->axis_ = 1;
  device::Tensor softmax_tensor(tensor->getDevice(), tensor->getDesc());
  base::Status status = op::softmax(tensor, op_param, &softmax_tensor);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("op::softmax failed!\n");
    return status;
  }
  float *data = (float *)softmax_tensor.getData();
  int batch = softmax_tensor.getShapeIndex(0);
  int num_classes = softmax_tensor.getShapeIndex(1);

  ClassificationResult *results = new ClassificationResult();
  param->topk_ = std::min(num_classes, param->topk_);
  int topk = param->topk_;
  results->labels_.resize(topk * batch);

  // 使用优先队列找出topk个最大值
  for (int b = 0; b < batch; ++b) {
    float *iter_data = data + b * num_classes;
    std::vector<int> label_ids_ = topKIndices(iter_data, num_classes, topk);

    for (int i = 0; i < topk; ++i) {
      results->labels_[i + b * topk].index_ = b;
      results->labels_[i + b * topk].label_ids_ = label_ids_[i];
      results->labels_[i + b * topk].scores_ = *(iter_data + label_ids_[i]);
    }
  }

  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  return base::kStatusCodeOk;
}

// Not recommended to use createGraph method anymore
dag::Graph *createClassificationResnetGraph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *infer_input = graph->createEdge("data");
  dag::Edge *infer_output = graph->createEdge("resnetv17_dense0_fwd");

  dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(
      "preprocess", input, infer_input);

  dag::Node *infer = graph->createInfer<infer::Infer>(
      "infer", inference_type, infer_input, infer_output);

  dag::Node *post = graph->createNode<ClassificationPostProcess>(
      "postprocess", infer_output, output);

  preprocess::CvtclorResizeParam *pre_param =
      dynamic_cast<preprocess::CvtclorResizeParam *>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 224;
  pre_param->w_ = 224;
  pre_param->mean_[0] = 0.485;
  pre_param->mean_[1] = 0.456;
  pre_param->mean_[2] = 0.406;
  pre_param->std_[0] = 0.229;
  pre_param->std_[1] = 0.224;
  pre_param->std_[2] = 0.225;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  ClassificationPostParam *post_param =
      dynamic_cast<ClassificationPostParam *>(post->getParam());
  post_param->topk_ = 1;

  return graph;
}
// Not recommended to use createGraph method anymore
dag::Graph *createClassificationResnetGraphV0(
    const std::string &name, dag::Edge *input, dag::Edge *output,
    base::Param *pre_param, base::Param *infer_param, base::Param
    *post_param) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Node *pre =
      graph->createNode<preprocess::CvtColorResize>("preprocess", input);
  dag::Node *infer = graph->createInfer<infer::Infer>("infer", pre);
  dag::Node *post =
  graph->createNode<ClassificationPostProcess>("postprocess",
                                                                 infer,
                                                                 output);
  pre->setParam(pre_param);
  infer->setParam(infer_param);
  post->setParam(post_param);

  return graph;
}
// Not recommended to use createGraph method anymore
dag::Graph *createClassificationResnetGraphOptInterface(
    const std::string &name, dag::Edge *input, dag::Edge *output,
    NodeDesc<preprocess::CvtColorResizeParam> *pre_desc,
    NodeDesc<infer::InferParam> *infer_desc,
    NodeDesc<ClassificationPostParam> *post_desc) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(pre_desc);
  dag::Node *infer =
  graph->createInfer<ClassificationResnetGraph>(infer_desc); dag::Node *post
  = graph->createNode<ClassificationPostProcess>(post_desc);

  graph->dump();

  return graph;
}

/**
 * @brief Static graph implementation for ResNet classification
 *
 * This class represents a fixed graph structure for image classification
 using
 * ResNet models. The graph consists of three main stages:
 * 1. Preprocessing: Converts input image format and resizes to model input
 size
 * 2. Inference: Runs the ResNet model inference
 * 3. Postprocessing: Processes model output to get classification results
 *
 * All node connections and data flow paths are determined at construction
 time
 * and cannot be modified during runtime. This provides better performance
 but
 * less flexibility compared to dynamic graphs.
 *
 * Expected data flow:
 * input image -> color conversion & resize -> ResNet inference ->
 * classification results
 */
class ClassificationResnetGraph : public dag::Graph {
 public:
  /**
   * @brief Constructor that builds the complete graph structure
   * @param name Graph name for identification
   * @param input Input edge that receives the raw image data
   *        Must be named "input" to match graph expectations
   * @param output Output edge that provides classification results
   *        Must be named "output" to match graph expectations
   * @param pre_desc Preprocess node descriptor defining color conversion and
   * resize inputs_: Must contain "input" edge name outputs_: Must contain
   * "data" edge name for model input Parameters control color space
   conversion
   * and resize operation
   * @param infer_desc Inference node descriptor for ResNet model
   *        inputs_: Must contain "data" edge name matching preprocess output
   *        outputs_: Must contain "resnetv17_dense0_fwd" edge name
   *        Parameters specify model configuration and runtime settings
   * @param post_desc Postprocess node descriptor for classification
   *        inputs_: Must contain "resnetv17_dense0_fwd" matching model
   output
   *        outputs_: Must contain "output" edge name
   *        Parameters control classification post-processing like top-k
   * @note The graph structure is fixed after construction
   */
  ClassificationResnetGraph(const std::string &name, dag::Edge *input,
                            dag::Edge *output,
                            preprocess::CvtColorResizeParam *pre_param,
                            infer::InferParam *infer_param,
                            ClassificationPostParam *post_param)
      : dag::Graph(name, input, output) {
    dag::Edge *infer_input = graph->createEdge("data");
    dag::Edge *infer_output = graph->createEdge("resnetv17_dense0_fwd");

    pre_ = graph->createNode<preprocess::CvtColorResize>(
        "preprocess", input, infer_input, pre_param);

    infer_ = graph->createInfer<infer::Infer>(
        "infer", inference_type, infer_input, infer_output, infer_param);

    post_ = graph->createNode<ClassificationPostProcess>(
        "postprocess", infer_output, output, post_param);

    // Print graph structure for debugging
    graph->dump();
  }

  /**
   * @brief Set preprocessing parameters
   * @param pixel_type Input image pixel format (e.g. RGB, BGR, etc)
   * @return kStatusCodeOk on success
   * @note This allows runtime configuration of input format while
   maintaining
   * graph structure
   */
  base::Status setPreParam(base::PixelType pixel_type) {
    preprocess::CvtColorResizeParam *param =
        dynamic_cast<preprocess::CvtColorResizeParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  /**
   * @brief Execute graph inference
   * @return kStatusCodeOk on success
   * @note Execution flow:
   * 1. Pre-run hook for any setup/debug/timeprofile
   * 2. Graph inference through all nodes
   * 3. Post-run hook for cleanup/debug/timeprofile
   */
  virtual base::Status run() {
    hook_prerun();
    dag::Graph::run();
    hook_postrun();
    return base::kStatusCodeOk;
  }

 private:
  // Pointer to preprocess node for color conversion and resize
  dag::Node *pre_ = nullptr;
  dag::Node *infer_ =
      nullptr;  // Pointer to inference node running ResNet model
  dag::Node *post_ =
      nullptr;  // Pointer to postprocess node for classification results
};

/**
 * @brief A static graph implementation for ResNet-based classification
 *
 * This class represents a fixed graph structure for image classification
 using
 * ResNet models. The graph consists of three main stages:
 * 1. Preprocessing: Converts input image format and resizes to model input
 size
 * 2. Inference: Runs the ResNet model inference
 * 3. Postprocessing: Processes model output to get classification results
 *
 * All node connections and data flow paths are determined at construction
 time
 * and cannot be modified during runtime. This provides better performance
 but
 * less flexibility compared to dynamic graphs.
 *
 * Expected data flow:
 * input image -> color conversion & resize -> ResNet inference ->
 * classification results
 */
class ClassificationResnetGraph : public dag::Graph {
 public:
  /**
   * @brief Constructor that builds the complete graph structure
   * @param name Graph name for identification
   * @param input Input edge that receives the raw image data
   *        Must be named "input" to match graph expectations
   * @param output Output edge that provides classification results
   *        Must be named "output" to match graph expectations
   * @param pre_desc Preprocess node descriptor defining color conversion and
   * resize inputs_: Must contain "input" edge name outputs_: Must contain
   * "data" edge name for model input Parameters control color space
   conversion
   * and resize operation
   * @param infer_desc Inference node descriptor for ResNet model
   *        inputs_: Must contain "data" edge name matching preprocess output
   *        outputs_: Must contain "resnetv17_dense0_fwd" edge name
   *        Parameters specify model configuration and runtime settings
   * @param post_desc Postprocess node descriptor for classification
   *        inputs_: Must contain "resnetv17_dense0_fwd" matching model
   output
   *        outputs_: Must contain "output" edge name
   *        Parameters control classification post-processing like top-k
   * @note The graph structure is fixed after construction
   */
  ClassificationResnetGraph(const std::string &name, dag::Edge *input,
                            dag::Edge *output,
                            NodeDesc<preprocess::CvtColorResizeParam>
                            *pre_desc, NodeDesc<infer::InferParam>
                            *infer_desc, NodeDesc<ClassificationPostParam>
                            *post_desc)
      : dag::Graph(name, input, output) {
    pre_ = graph->createNode<preprocess::CvtColorResize>(pre_desc);
    infer_ = graph->createInfer<infer::Infer>(infer_desc);
    post_ = graph->createNode<ClassificationPostProcess>(post_desc);
  }

  /**
   * @brief Set preprocessing parameters
   * @param pixel_type Input image pixel format (e.g. RGB, BGR, etc)
   * @return kStatusCodeOk on success
   * @note This allows runtime configuration of input format while
   maintaining
   * graph structure
   */
  base::Status setPreParam(base::PixelType pixel_type) {
    preprocess::CvtColorResizeParam *param =
        dynamic_cast<preprocess::CvtColorResizeParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  /**
   * @brief Execute graph inference
   * @return kStatusCodeOk on success
   * @note Execution flow:
   * 1. Pre-run hook for any setup/debug/timeprofile
   * 2. Graph inference through all nodes
   * 3. Post-run hook for cleanup/debug/timeprofile
   */
  virtual base::Status run() {
    hook_prerun();
    dag::Graph::run();
    hook_postrun();
    return base::kStatusCodeOk;
  }

 private:
  // Pointer to preprocess node for color conversion and resize
  dag::Node *pre_ = nullptr;
  dag::Node *infer_ =
      nullptr;  // Pointer to inference node running ResNet model
  dag::Node *post_ =
      nullptr;  // Pointer to postprocess node for classification results
};

/**
 * @brief Implementation of ResNet classification network graph structure
 * @details This class sits between static and dynamic graphs, with each desc
 * specifying outputs_ Contains three main nodes:
 * 1. Preprocessing node (pre_): Performs image color conversion and resizing
 * 2. Inference node (infer_): Executes ResNet model inference
 * 3. Postprocessing node (post_): Processes classification results
 */
class ClassificationResnetGraph : public dag::Graph {
 public:
  /**
   * @brief Constructor to create and initialize graph structure
   * @param name Graph name
   * @param pre_desc Preprocessing node descriptor containing output names
   * @param infer_desc Inference node descriptor containing output names
   * @param post_desc Postprocessing node descriptor containing output names
   */
  ClassificationResnetGraph(const std::string &name,
                            NodeDesc<preprocess::CvtColorResizeParam>
                            *pre_desc, NodeDesc<infer::InferParam>
                            *infer_desc, NodeDesc<ClassificationPostParam>
                            *post_desc)
      : dag::Graph(name) {
    // Create preprocessing node for image preprocessing
    pre_ = graph->createNode<preprocess::CvtColorResize>(pre_desc);
    // Create inference node for ResNet model execution
    infer_ = graph->createInfer<ClassificationResnetGraph>(infer_desc);
    // Create postprocessing node for classification results
    post_ = graph->createNode<ClassificationPostProcess>(post_desc);
  }

  /**
   * @brief Set preprocessing parameters
   * @param pixel_type Input image pixel format (e.g. RGB, BGR)
   * @return kStatusCodeOk on success
   */
  base::Status setPreParam(base::PixelType pixel_type) {
    preprocess::CvtColorResizeParam *param =
        dynamic_cast<preprocess::CvtColorResizeParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  /**
   * @brief Execute graph inference
   * @return kStatusCodeOk on success
   * @details Execution flow:
   * 1. Call pre-run hook
   * 2. Execute graph inference
   * 3. Call post-run hook
   */
  virtual base::Status run() {
    hook_prerun();
    dag::Graph::run();
    hook_postrun();
    return base::kStatusCodeOk;
  }

  /**
   * @brief Forward propagation function
   * @param inputs Input tensor list
   * @param output_names List of specified output names
   * @return Output tensor list
   * @details Execution order:
   * 1. Preprocessing
   * 2. Model inference
   * 3. Postprocessing
   */
  std::vector<dag::Edge *> forward(
      std::vector<dag::Edge *> inputs,
      std::initializer_list<const std::string &> output_names = {}) {
    std::vector<dag::Edge *> outputs = dag::forward(inputs, output_names);

    inputs = pre_(inputs);    // Output names determined by pre_desc
    inputs = infer_(inputs);  // Output names determined by infer_desc
    outputs = post_(inputs);  // Output names determined by infer_desc
    return outputs;
  }

 private:
  dag::Node *pre_;    ///< Preprocessing node pointer
  dag::Node *infer_;  ///< Inference node pointer
  dag::Node *post_;   ///< Postprocessing node pointer
};

/**
 * @brief Dynamic graph implementation of ResNet classification network
 * @details Dynamic graph containing three nodes: preprocessing, inference
 and
 * postprocessing Preprocessing node handles image format conversion and
 * resizing Inference node executes ResNet model inference Postprocessing
 node
 * processes inference results, including softmax and top-k classification
 */
class ClassificationResnetGraph : public dag::Graph {
 public:
  /**
   * @brief Constructor
   * @param name Graph name
   * @param pre_param Preprocessing parameters, including input/output image
   * format, size etc.
   * @param infer_param Inference parameters, including model path, device
   type
   * etc.
   * @param post_param Postprocessing parameters, including number of
   classes,
   * thresholds etc.
   */
  ClassificationResnetGraph(const std::string &name,
                            preprocess::CvtColorResizeParam *pre_param,
                            infer::InferParam *infer_param,
                            ClassificationPostParam *post_param)
      : dag::Graph(name) {
    pre_ = graph->createNode<preprocess::CvtColorResize>(pre_param);
    infer_ = graph->createInfer<infer::Infer>(infer_param);
    post_ = graph->createNode<ClassificationPostProcess>(post_param);
  }

  /**
   * @brief Set preprocessing parameters
   * @param pixel_type Input image pixel type, supports RGB, BGR, RGBA etc.
   * @return kStatusCodeOk on success
   */
  base::Status setPreParam(base::PixelType pixel_type) {
    preprocess::CvtColorResizeParam *param =
        dynamic_cast<preprocess::CvtColorResizeParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  /**
   * @brief Execute graph inference
   * @return kStatusCodeOk on success
   * @details Execution flow:
   * 1. Call pre-run hook for setup
   * 2. Execute graph inference in topological order
   * 3. Call post-run hook for cleanup
   */
  virtual base::Status run() {
    hook_prerun();
    dag::Graph::run();
    hook_postrun();
    return base::kStatusCodeOk;
  }

  /**
   * @brief Forward propagation function
   * @param inputs Input tensor list containing image data
   * @param output_names List of specified output names for intermediate
   results
   * @return Output tensor list containing classification results
   * @details Execution order:
   * 1. Preprocessing: Convert input image to model required format
   * 2. Model inference: Execute ResNet model computation
   * 3. Postprocessing: Process model output to get final classification
   results
   */
  std::vector<dag::Edge *> forward(
      std::vector<dag::Edge *> inputs,
      std::initializer_list<const std::string &> output_names = {}) {
    std::vector<dag::Edge *> outputs = dag::forward(inputs, output_names);

    // Preprocessing output "data" represents processed image
    inputs = pre_(inputs, {"data"});
    // Inference output "resnetv17_dense0_fwd"
    // is the final FC layer output
    inputs = infer_(inputs, {"resnetv17_dense0_fwd"});
    // Postprocessing outputs specified by output_names,
    // typically class results and confidence
    outputs = post_(inputs, output_names);
    return outputs;
  }

 private:
  dag::Node *pre_;    ///< Preprocessing node pointer for image preprocessing
  dag::Node *infer_;  ///< Inference node pointer for model inference
  dag::Node *post_;   ///< Postprocessing node pointer for result processing
};

}  // namespace classification
}  // namespace nndeploy
```