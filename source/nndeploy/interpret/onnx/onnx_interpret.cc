

#include "nndeploy/interpret/onnx/onnx_interpret.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

std::shared_ptr<op::OpDesc> OnnxInterpret::convertOnnxNodeToOpDesc(
    const onnx::NodeProto& onnx_node) {
  std::string name = onnx_node.name();
  op::OpType op_type = convertOnnxOpTypeToOpType(onnx_node.op_type());
  std::vector<std::string> inputs;
  for (int i = 0; i < onnx_node.input_size(); ++i) {
    inputs.push_back(onnx_node.input(i));
  }
  std::vector<std::string> outputs;
  for (int i = 0; i < onnx_node.output_size(); ++i) {
    outputs.push_back(onnx_node.output(i));
  }
  std::shared_ptr<op::OpDesc> op_desc =
      std::make_shared<op::OpDesc>(name, op_type, inputs, outputs);
  convert = createParamConverter(op_type);
  return op_desc;
}
device::Tensor* OnnxInterpret::convertOnnxInitializerToTensor(
    const onnx::TensorProto& initializer) {
  // tensor->name_ = initializer.name();
  // tensor->data_type_ =
  // convertOnnxDataTypeToDataType(initializer.data_type()); tensor->dims_ =
  // convertOnnxDimsToDims(initializer.dims()); tensor->data_ =
  // convertOnnxDataToData(initializer);
  device::Tensor* tensor = new device::Tensor();
  return tensor;
}
std::shared_ptr<ValueDesc> OnnxInterpret::convertOnnxValueInfoToValueDesc(
    const onnx::ValueInfoProto& input) {
  std::string name = input.name();
  DataType data_type =
      convertOnnxDataTypeToDataType(input.type().tensor_type().elem_type());
  Dims dims = convertOnnxDimsToDims(input.type().tensor_type().shape().dim());
  return std::make_shared<ValueDesc>(name, data_type, dims);
}

base::Status OnnxInterpret::interpret(
    const std::vector<std::string>& model_value,
    const std::vector<op::ValueDesc>& input) {
  base::Status status = base::kStatusCodeOk;

  // 读模型文件
  std::ifstream input_stream(model_value[0],
                             std::ifstream::in | std::ifstream::binary);
  if (!input_stream.is_open()) {
    NNDEPLOY_LOGE("model_value[%s] is error.\n", model_value[0].c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  google::protobuf::io::IstreamInputStream proto_input_stream(&input_stream);
  google::protobuf::io::CodedInputStream coded_input_stream(
      &proto_input_stream);
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  coded_input_stream.SetTotalBytesLimit(INT_MAX);
#else
  coded_input_stream.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif
  this->onnx_model_ = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());
  bool success = this->onnx_model_->ParseFromCodedStream(&coded_input_stream);
  input_stream.close();

  // 解析图
  const auto& onnx_graph = onnx_model_->graph();
  // # 解析节点
  const int node_size = onnx_graph->node_size();
  for (int i = 0; i < node_size; ++i) {
    const auto& onnx_node = onnx_graph->node(i);
    std::shared_ptr<op::OpDesc> op_desc = convertOnnxNodeToOpDesc(onnx_node);
    model_desc_->op_descs_.push_back(op_desc);
  }
  // # 解析权重
  const int initializer_size = onnx_graph->initializer_size();
  for (int i = 0; i < initializer_size; ++i) {
    const auto& initializer = onnx_graph->initializer(i);
    std::string name = initializer.name();
    // 前拷贝权重数据
    device::Tensor* tensor = convertOnnxInitializerToTensor(initializer);
    model_desc_->weights_.insert(std::make_pair(name, tensor));
  }
  // # 解析输入
  const int input_size = onnx_graph->input_size();
  for (int i = 0; i < input_size; ++i) {
    const auto& input = onnx_graph->input(i);
    std::shared_ptr<ValueDesc> value_desc =
        convertOnnxValueInfoToValueDesc(input);
    model_desc_->inputs_.insert(value_desc);
  }
  // 解析输出
  const int output_size = onnx_graph->output_size();
  for (int i = 0; i < output_size; ++i) {
    const auto& output = onnx_graph->output(i);
    std::shared_ptr<op::ValueDesc> value_desc =
        convertOnnxValueInfoToValueDesc(output);
    model_desc_->outputs_.insert(value_desc);
  }

  return status;
}

}  // namespace interpret
}  // namespace nndeploy
