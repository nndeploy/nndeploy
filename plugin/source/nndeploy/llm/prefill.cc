#include "nndeploy/llm/prefill.h"

namespace nndeploy {
namespace llm {

Prefill::Prefill(const std::string& name, std::vector<dag::Edge*> inputs,
                 std::vector<dag::Edge*> outputs)
    : dag::Graph(name, inputs, outputs) {
  // TODO: 初始化预填充推理图
  // 1. 创建内部节点
  // 2. 连接节点之间的边
  // 3. 设置图的输入输出
}

Prefill::~Prefill() {
  // TODO: 清理资源
  // 1. 释放内部节点资源
  // 2. 清理边连接
}

base::Status Prefill::defaultParam() {
  // TODO: 设置默认参数
  // 1. 设置tokenizer参数
  // 2. 设置embedding参数
  // 3. 设置推理参数
  // 4. 设置采样参数
  return base::kStatusCodeOk;
}

base::Status Prefill::make(const dag::NodeDesc& tokenizer,
                           const dag::NodeDesc& embedding,
                           const dag::NodeDesc& infer,
                           const dag::NodeDesc& sample) {
  // TODO: 实现make逻辑
  // 1. 设置tokenizer参数
  // 2. 设置embedding参数
  // 3. 设置推理参数
  // 4. 设置采样参数
  return base::kStatusCodeOk;
}

base::Status Prefill::serialize(rapidjson::Value& json,
                                rapidjson::Document::AllocatorType& allocator) {
  // TODO: 实现序列化
  // 1. 调用父类序列化方法
  // 2. 序列化特有的参数
  // 3. 序列化内部节点配置
  return dag::Graph::serialize(json, allocator);
}

base::Status Prefill::deserialize(rapidjson::Value& json) {
  // TODO: 实现反序列化
  // 1. 调用父类反序列化方法
  // 2. 反序列化特有的参数
  // 3. 重建内部节点配置
  return dag::Graph::deserialize(json);
}

std::vector<dag::Edge*> Prefill::forward(dag::Edge* input) {
  // TODO: 实现前向推理逻辑
  // 1. 输入文本通过tokenizer节点进行分词
  // 2. token通过embedding节点转换为向量
  // 3. embedding向量通过推理节点进行预填充推理
  // 4. 推理结果通过采样节点生成输出token
  // 5. 返回输出边集合
  std::vector<dag::Edge *> outputs;
  return outputs;
}

REGISTER_NODE("nndeploy::llm::Prefill", Prefill);

}  // namespace llm
}  // namespace nndeploy