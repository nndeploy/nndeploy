#include <memory>

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/net/net.h"
#include "nndeploy/op/expr.h"

using namespace nndeploy;

// TODO:
// [ ] init with config ?
// [x] embedding op
// [ ] input should be dynamic shape
// [x] position_ids and attention_mask how to fix, as input
// [ ] rope op
//    apply rotary position embedding
//    [x] slice
//    [x] reshape
//    [x] broadcast
//    [x] transpose
//    [x] concat
//    [x] cos
//    [x] sin
// [x] act silu
// [x] cast
//    [x] gemm

// [ ] suppose not getShape
// [ ] rope big op
// [ ] reshape, transpose use weight, make weight
// [ ] memory opt


#if 0
llama2 configs
LlamaConfig {
  "_name_or_path": "NousResearch/Llama-2-7b-chat-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.42.3",
  "use_cache": true,
  "vocab_size": 32000
}
#endif

// now make feedforward op
// clang-format off
//     def forward(self, x: torch.Tensor):
//         # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
//         swish = F.silu(self.w1(x))
//         # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
//         x_V = self.w3(x)
//         # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
//         x = swish * x_V
//         # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
//         return x
//         x = self.w2(x)
// clang-format on

std::shared_ptr<op::Expr> makeMLP(ir::ModelDesc *model_desc,
                                  std::shared_ptr<op::Expr> input,
                                  std::string weight_gate_proj,
                                  std::string weight_up_proj,
                                  std::string weight_down_proj) {
  auto gate_proj = makeMatmul(model_desc, input, weight_gate_proj, "");
  auto up_proj = makeMatmul(model_desc, input, weight_up_proj, "");
  auto act_out = makeSilu(model_desc, gate_proj);
  auto mul_out = makeMul(model_desc, act_out, up_proj);
  auto down_proj = makeMatmul(model_desc, mul_out, weight_down_proj, "");
  return down_proj;
}

std::shared_ptr<op::Expr> makeDecorderLayerNorm(ir::ModelDesc *model_desc,
                                                std::shared_ptr<op::Expr> input,
                                                std::string weight,
                                                std::string residual,
                                                std::vector<int> norm_shape,
                                                double eps) {
  auto param = std::make_shared<ir::RMSNormParam>(norm_shape, eps);
  auto norm = makeRMSNorm(model_desc, input, param, weight, "");
  return norm;
}

// inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
// dtype=torch.int64).float().to(device) / self.dim))

// [ ] how to get cos sin
std::pair<std::shared_ptr<op::Expr>, std::shared_ptr<op::Expr>>
makeRotaryEmbedding(std::shared_ptr<op::Expr> input_ids, std::string inv_freq) {
  // int32_t vec_size = head_dim / 2;
  // int32_t *inv_freq_data = (int32_t *)malloc(sizeof(int32_t) * vec_size);
  // for (int i = 0; i < head_dim; i += 2) {
  //   int32_t x = i / head_dim;
  //   inv_freq_data[int(i / 2)] = 1 / std::pow(base, x);
  // }
  return std::pair<std::shared_ptr<op::Expr>, std::shared_ptr<op::Expr>>(
      std::make_shared<op::Expr>("cos"), std::make_shared<op::Expr>("sin"));
}

std::shared_ptr<op::Expr> makeLlamaAttention(
    ir::ModelDesc *model_desc, std::shared_ptr<op::Expr> hidden_states,
    std::shared_ptr<op::Expr> input_ids, std::string weight_q_proj,
    std::string weight_k_proj, std::string weight_v_proj,
    std::string weight_out_proj, std::string rope_inv_freq) {
  // qkv gen
  auto query_states = makeMatmul(model_desc, hidden_states, weight_q_proj, "");
  auto key_states = makeMatmul(model_desc, hidden_states, weight_k_proj, "");
  auto value_states = makeMatmul(model_desc, hidden_states, weight_v_proj, "");

  // reshape transpose before qkv
  auto reshape_param = std::make_shared<ir::ReshapeParam>();
  reshape_param->allowzero_ = 1;

  auto q_reshape =
      op::makeInput(model_desc, "q_reshape", base::dataTypeOf<int32_t>());
  auto kv_reshape =
      op::makeInput(model_desc, "q_reshape", base::dataTypeOf<int32_t>());
  auto qkv_transpose =
      op::makeInput(model_desc, "qkv_transpose", base::dataTypeOf<int32_t>());

  query_states =
      makeReshape(model_desc, query_states, q_reshape, reshape_param);
  key_states = makeReshape(model_desc, key_states, kv_reshape, reshape_param);
  value_states =
      makeReshape(model_desc, value_states, kv_reshape, reshape_param);

  query_states = makeTranspose(model_desc, query_states, qkv_transpose);
  key_states = makeTranspose(model_desc, key_states, qkv_transpose);
  value_states = makeTranspose(model_desc, value_states, qkv_transpose);

  std::shared_ptr<op::Expr> cos, sin;
  std::tie(cos, sin) = makeRotaryEmbedding(input_ids, rope_inv_freq);
}

class Llama2DecorderLayers : public ir::ModelDesc {
 public:
  Llama2DecorderLayers(int layer) : layer_no(layer) {
    std::string layer_str = std::to_string(layer_no);
    input_norm_weight = "model.layers." + layer_str + ".input_layernorm.weight";
    post_norm_weight =
        "model.layers." + layer_str + ".post_attention_layernorm.weight";
    mlp_down_proj_weight =
        "model.layers." + layer_str + ".mlp.down_proj.weight";
    mlp_gate_proj_weight =
        "model.layers." + layer_str + ".mlp.gate_proj.weight";
    mlp_up_proj_weight = "model.layers." + layer_str + ".mlp.up_proj.weight";
    attn_k_proj_weight =
        "model.layers." + layer_str + ".self_attn.k_proj.weight";
    attn_q_proj_weight =
        "model.layers." + layer_str + ".self_attn.q_proj.weight";
    attn_v_proj_weight =
        "model.layers." + layer_str + ".self_attn.v_proj.weight";
    attn_o_proj_weight =
        "model.layers." + layer_str + ".self_attn.o_proj.weight";
    attn_rope_inv_freq_weight =
        "model.layers." + layer_str + ".self_attn.rotary_emb.inv_freq";
  };
  ~Llama2DecorderLayers(){};

  void init() {
    auto hidden_states =
        op::makeInput(this, "hidden_states", base::dataTypeOf<float>(), {});
    auto attention_mask =
        op::makeInput(this, "attention_mask", base::dataTypeOf<float>(), {});
    auto postion_ids =
        op::makeInput(this, "position_ids", base::dataTypeOf<int32_t>(), {});

    // ? how to fix this
    auto postion_ids_shape = op::makeInput(this, "position_ids_shape",
                                           base::dataTypeOf<int32_t>(), {});

    // auto cos = op::makeInput(this, "cos", base::dataTypeOf<float>(), {});
    // auto sin = op::makeInput(this, "sin", base::dataTypeOf<float>(), {});
    auto input_norm = makeDecorderLayerNorm(
        this, hidden_states, input_norm_weight, "", {4096}, 1e-5);
  }

 private:
  int layer_no{-1};
  std::string input_norm_weight;
  std::string mlp_down_proj_weight;
  std::string mlp_up_proj_weight;
  std::string mlp_gate_proj_weight;
  std::string post_norm_weight;
  std::string attn_k_proj_weight;
  std::string attn_q_proj_weight;
  std::string attn_v_proj_weight;
  std::string attn_o_proj_weight;
  std::string attn_rope_inv_freq_weight;
  // std::string mlp_down_proj_weight;
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  {
    // std::string weight_path_1 =
    //     "/home/samsung_1tb/realtyxxx/.cache/huggingface/hub/"
    //     "models--NousResearch--Llama-2-7b-chat-hf/snapshots/"
    //     "351844e75ed0bcbbe3f10671b3c808d2b83894ee/"
    //     "model-00001-of-00002.safetensors";
    // std::string weight_path_2 =
    //     "/home/samsung_1tb/realtyxxx/.cache/huggingface/hub/"
    //     "models--NousResearch--Llama-2-7b-chat-hf/snapshots/"
    //     "351844e75ed0bcbbe3f10671b3c808d2b83894ee/"
    //     "model-00002-of-00002.safetensors";

    std::string weight_path =
        "/home/samsung_1tb/realtyxxx/github/total.safetensors";

    ir::Interpret *interpret = ir::createInterpret(base::kModelTypeDefault);
    // interpret->interpret({"", weight_path_1});
    // interpret->interpret({"", weight_path_2});
    interpret->interpret({"", weight_path});

    std::cout << "\n"
              << "from load" << std::endl;
    interpret->model_desc_->dump(std::cout);

    // auto &weights = interpret->model_desc_->weights_;
    // for (auto weight : weights) {
    // std::cout << weight.first << " " << weight.second << std::endl;
    // std::cout << weight.first << std::endl;
    // }

    // nndeploy::net::Net *llama2 = new nndeploy::net::Net();
    // llama2->setInterpret(interpret);

    // interpret->saveModelToFile("", weight_path);

    delete interpret;
  }

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}
