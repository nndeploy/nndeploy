
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

  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::classification::ClassificationPostProcess",
              ClassificationPostProcess);
REGISTER_NODE("nndeploy::classification::ClassificationResnetGraph",
              ClassificationResnetGraph);

}  // namespace classification
}  // namespace nndeploy