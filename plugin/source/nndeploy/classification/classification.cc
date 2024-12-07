
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

dag::TypeGraphRegister g_register_resnet_graph(NNDEPLOY_RESNET,
                                               createClassificationResnetGraph);

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
    ;
    for (int i = 0; i < topk; ++i) {
      results->labels_[i + b * topk].index_ = b;
      results->labels_[i + b * topk].label_ids_ = label_ids_[i];
      results->labels_[i + b * topk].scores_ = *(iter_data + label_ids_[i]);
    }
  }

  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  return base::kStatusCodeOk;
}

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

}  // namespace classification
}  // namespace nndeploy
