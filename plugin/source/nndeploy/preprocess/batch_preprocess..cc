
#include "nndeploy/preprocess/batch_preprocess.h"

#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

base::Status BatchPreprocess::setNodeKey(const std::string &key) {
  node_key_ = key;
  return base::kStatusCodeOk;
}

base::Status BatchPreprocess::make() {
  std::vector<std::string> input_names = this->getInputNames();
  std::vector<std::string> output_names = this->getRealOutputsName();
  dag::NodeDesc desc(node_key_, "inner_preprocess_node", input_names,
                     output_names);
  node_ = this->createNode(desc);
  if (!node_) {
    NNDEPLOY_LOGE("Node creation failed for node_key: %s\n", node_key_.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  if (node_->getInputTypeInfo() != this->getInputTypeInfo() ||
      node_->getOutputTypeInfo() != this->getOutputTypeInfo()) {
    NNDEPLOY_LOGE(
        "Type mismatch: Node input/output types do not match BatchPreprocess "
        "types.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  return base::kStatusCodeOk;
}

base::Status BatchPreprocess::run() {
  std::vector<cv::Mat> *input_data =
      inputs_[0]->getGraphOutputAny<std::vector<cv::Mat>>();
  int batch_size = input_data->size();
  device::Tensor *dst_tensor = nullptr;
  for (int i = 0; i < batch_size; i++) {
    node_->run();
    dag::Edge *output = node_->getOutput();
    device::Tensor *single_tensor = output->getTensor(node_);
    if (single_tensor == nullptr) {
      NNDEPLOY_LOGE("single_tensor is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
    device::Device *device = single_tensor->getDevice();
    device::TensorDesc desc = single_tensor->getDesc();
    if (i == 0) {
      if (data_format_ == base::kDataFormatNDCHW ||
          data_format_ == base::kDataFormatNDHWC) {
        // 在这里，`desc.shape_`是一个表示张量形状的向量。`insert`函数用于在向量的指定位置插入一个元素。
        // `desc.shape_.begin() + 1`表示在向量的第二个位置插入元素。
        // `batch_size`是要插入的元素，表示批处理的大小。
        // 这行代码的作用是将批处理大小插入到张量形状的第二个位置，从而调整张量的形状以适应批处理。
        desc.shape_.insert(desc.shape_.begin() + 1, batch_size);
        desc.data_format_ = data_format_;
      } else {
        desc.shape_[0] = batch_size;
      }
      dst_tensor = outputs_[0]->create(device, desc);
    }
    void *single_data = single_tensor->getData();
    void *data = ((char *)dst_tensor->getData()) + single_tensor->getSize() * i;
    device->copy(data, single_data, single_tensor->getSize());
  }
  outputs_[0]->notifyWritten(dst_tensor);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::preprocess::BatchPreprocess", BatchPreprocess);

}  // namespace preprocess
}  // namespace nndeploy
