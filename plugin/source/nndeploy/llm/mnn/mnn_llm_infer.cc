// 封装MNN LLM推理引擎

/**
 * @brief MNN LLM推理引擎封装
 *
 * 基于MNN框架实现的大语言模型推理引擎，提供完整的LLM推理能力
 * 参考MNN transformers/llm/engine实现
 *
 *
 * 支持特性：
 * - 量化推理（INT4/INT8）
 * - 动态批处理
 * - KV缓存优化
 * - 流式生成
 */

#include "nndeploy/llm/mnn/mnn_llm_infer.h"

namespace nndeploy {
namespace llm {

REGISTER_NODE("nndeploy::llm::MnnLlmInfer", MnnLlmInfer);

REGISTER_LLM_INFER("MnnLlmInfer", "Qwen", MnnLlmInfer);

}  // namespace llm
}  // namespace nndeploy