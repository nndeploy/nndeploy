#include "nndeploy/llm/decode.h"

namespace nndeploy {
namespace llm {

Decode::Decode(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
    : dag::Loop(name, inputs, outputs),
      decode_token_node_(nullptr),
      decode_embedding_node_(nullptr),
      decode_infer_node_(nullptr),
      decode_sample_node_(nullptr) {
  // TODO: 初始化解码循环图
  // 1. 创建内部节点
  // 2. 连接节点之间的边
  // 3. 设置循环图的输入输出
}

Decode::~Decode() {
  // TODO: 清理资源
  // 1. 释放内部节点资源
  // 2. 清理边连接
}

base::Status Decode::defaultParam() {
  // TODO: 设置默认参数
  // 1. 设置tokenizer参数
  // 2. 设置embedding参数
  // 3. 设置推理参数
  // 4. 设置采样参数
  return base::kStatusCodeOk;
}

base::Status Decode::make(const dag::NodeDesc& tokenizer,
                          const dag::NodeDesc& embedding,
                          const dag::NodeDesc& infer,
                          const dag::NodeDesc& sample) {
  // TODO: 实现make逻辑
  // 1. 根据NodeDesc创建tokenizer节点
  // 2. 根据NodeDesc创建embedding节点
  // 3. 根据NodeDesc创建推理节点
  // 4. 根据NodeDesc创建采样节点
  // 5. 连接各节点形成循环处理流程
  return base::kStatusCodeOk;
}

base::Status Decode::serialize(rapidjson::Value& json,
                               rapidjson::Document::AllocatorType& allocator) {
  // TODO: 实现序列化
  // 1. 调用父类序列化方法
  // 2. 序列化特有的参数
  // 3. 序列化内部节点配置
  return dag::Loop::serialize(json, allocator);
}

base::Status Decode::deserialize(rapidjson::Value& json) {
  // TODO: 实现反序列化
  // 1. 调用父类反序列化方法
  // 2. 反序列化特有的参数
  // 3. 重建内部节点配置
  return dag::Loop::deserialize(json);
}

int Decode::loops() {
  // TODO: 实现循环次数逻辑
  // 1. 根据生成策略确定循环次数
  // 2. 可以是固定次数或基于条件判断
  // 3. 返回需要执行的循环次数
  return 1;  // 临时返回值，实际应根据业务逻辑确定
}

REGISTER_NODE("nndeploy::llm::Decode", Decode);

}  // namespace llm
}  // namespace nndeploy
