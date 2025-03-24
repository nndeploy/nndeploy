
// #include "nndeploy/stable_diffusion/clip.h"

// #include "nndeploy/infer/infer.h"
// #include "nndeploy/preprocess/convert_to.h"
// #include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

// namespace nndeploy {
// namespace stable_diffusion {

// class NNDEPLOY_CC_API TokenizerConcat : public dag::Node {
//  public:
//   TokenizerConcat(const std::string &name,
//                   std::initializer_list<dag::Edge *> inputs,
//                   std::initializer_list<dag::Edge *> outputs)
//       : dag::Node(name, inputs, outputs) {
//     param_ = std::make_shared<tokenizer::TokenizerPraram>();
//   }
//   virtual ~TokenizerConcat() {}

//   base::Status init() {
//     base::Status status = base::kStatusCodeOk;

//     // param_
//     tokenizer::TokenizerPraram *tokenizer_param =
//         (tokenizer::TokenizerPraram *)(param_.get());

//     if (tokenizer_param->tokenizer_type_ ==
//         tokenizer::TokenizerType::kTokenizerTypeHF) {
//       if (tokenizer_param->json_blob_.empty()) {
//         NNDEPLOY_LOGE("json_blob_ is empty\n");
//         return base::kStatusCodeErrorInvalidParam;
//       }
//       // Read blob from file.
//       std::string blob;
//       if (tokenizer_param->is_path_) {
//         blob = base::openFile(tokenizer_param->json_blob_);
//       } else {
//         blob = tokenizer_param->json_blob_;
//       }
//       tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
//     } else if (tokenizer_param->tokenizer_type_ ==
//                tokenizer::TokenizerType::kTokenizerTypeBPE) {
//       if (tokenizer_param->vocab_blob_.empty() ||
//           tokenizer_param->merges_blob_.empty()) {
//         NNDEPLOY_LOGE("vocab_blob_ or  merges_blob_ is empty\n");
//         return base::kStatusCodeErrorInvalidParam;
//       }
//       // Read blob from file.
//       std::string vocab_blob;
//       std::string merges_blob;
//       std::string added_tokens;
//       if (tokenizer_param->is_path_) {
//         vocab_blob = base::openFile(tokenizer_param->vocab_blob_);
//         merges_blob = base::openFile(tokenizer_param->merges_blob_);
//         added_tokens = base::openFile(tokenizer_param->added_tokens_);
//       } else {
//         vocab_blob = tokenizer_param->vocab_blob_;
//         merges_blob = tokenizer_param->merges_blob_;
//         added_tokens = tokenizer_param->added_tokens_;
//       }
//       tokenizer_ = tokenizers::Tokenizer::FromBlobByteLevelBPE(
//           vocab_blob, merges_blob, added_tokens);
//     } else if (tokenizer_param->tokenizer_type_ ==
//                tokenizer::TokenizerType::kTokenizerTypeSentencePiece) {
//       if (tokenizer_param->model_blob_.empty()) {
//         NNDEPLOY_LOGE("model_blob_ is empty\n");
//         return base::kStatusCodeErrorInvalidParam;
//       }
//       // Read blob from file.
//       std::string blob;
//       if (tokenizer_param->is_path_) {
//         blob = base::openFile(tokenizer_param->model_blob_);
//       } else {
//         blob = tokenizer_param->model_blob_;
//       }
//       tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
//     } else if (tokenizer_param->tokenizer_type_ ==
//                tokenizer::TokenizerType::kTokenizerTypeRWKVWorld) {
//       if (tokenizer_param->model_blob_.empty()) {
//         NNDEPLOY_LOGE("model_blob_ is empty\n");
//         return base::kStatusCodeErrorInvalidParam;
//       }
//       // Read blob from file.
//       std::string blob;
//       if (tokenizer_param->is_path_) {
//         // blob = base::openFile(tokenizer_param->model_blob_);
//         blob = tokenizer_param->model_blob_;
//       } else {
//         NNDEPLOY_LOGE("model_blob_ is in-memory\n");
//         return base::kStatusCodeErrorInvalidParam;
//       }
//       tokenizer_ = tokenizers::Tokenizer::FromBlobRWKVWorld(blob);
//     } else {
//       status = base::kStatusCodeErrorInvalidParam;
//       NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
//                              "Invalid tokenizer type!");
//     }

//     return status;
//   }

//   base::Status deinit() {
//     base::Status status = base::kStatusCodeOk;
//     return status;
//   }

//   virtual base::Status run() {
//     base::Status status = base::kStatusCodeOk;

//     // param_
//     tokenizer::TokenizerPraram *tokenizer_concat_param =
//         (tokenizer::TokenizerPraram *)(param_.get());

//     tokenizer::TokenizerText *text_param1 =
//         (tokenizer::TokenizerText *)(inputs_[0]->getParam(this));
//     tokenizer::TokenizerText *text_param2 =
//         (tokenizer::TokenizerText *)(inputs_[1]->getParam(this));
//     int index = inputs_[0]->getIndex(this);

//     std::vector<std::string> texts;
//     texts.insert(texts.end(), text_param1->texts_.begin(),
//                  text_param1->texts_.end());
//     texts.insert(texts.end(), text_param2->texts_.begin(),
//                  text_param2->texts_.end());

//     std::vector<std::vector<int32_t>> ids;
//     ids.reserve(texts.size());
//     for (const auto &text : texts) {
//       // NNDEPLOY_LOGE("text=%s\n", text.c_str());
//       ids.push_back(tokenizer_->Encode(text));
//     }
//     // for (size_t i = 0; i < ids.size(); ++i) {
//     //   printEncodeResult(ids[i]);
//     // }

//     device::Device *device = device::getDevice(device_type_);
//     if (device == nullptr) {
//       NNDEPLOY_LOGE("device is nullptr\n");
//       return base::kStatusCodeErrorInvalidParam;
//     }
//     int32_t batch = (int32_t)(texts.size());
//     device::TensorDesc desc(base::dataTypeOf<int32_t>(), base::kDataFormatNC,
//                             {batch, tokenizer_concat_param->max_length_});
//     device::Tensor *tensor = outputs_[0]->create(device, desc, index);
//     tensor->set(49407);

//     int32_t *value = (int32_t *)tensor->getData();
//     for (int j = 0; j < batch; j++) {
//       int len = ids[j].size() > tokenizer_concat_param->max_length_
//                     ? tokenizer_concat_param->max_length_
//                     : ids[j].size();
//       value += j * tokenizer_concat_param->max_length_;
//       for (int i = 0; i < len; i++) {
//         value[i] = ids[j][i];
//       }
//     }

//     outputs_[0]->notifyWritten(tensor);
//     return status;
//   }

//   void printEncodeResult(const std::vector<int> &ids) {
//     std::cout << "tokens=[";
//     for (size_t i = 0; i < ids.size(); ++i) {
//       if (i != 0) std::cout << ", ";
//       std::cout << ids[i];
//     }
//     std::cout << "]" << std::endl;
//   }

//  private:
//   std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
// };

// dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *prompt,
//                             dag::Edge *negative_prompt, dag::Edge *output,
//                             base::InferenceType inference_type,
//                             std::vector<base::Param *> &param) {
//   dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt},
//   {output});

//   /**
//    * @brief tokenizer_concat
//    */
//   dag::Edge *input_ids = graph->createEdge("input_ids");
//   dag::Node *tokenizer_concat = graph->createNode<TokenizerConcat>(
//       "tokenizer_concat", {prompt, negative_prompt}, {input_ids});
//   tokenizer_concat->setParam(param[0]);

//   /**
//    * @brief createInfer
//    * 多batch的推理
//    */
//   dag::Node *infer = graph->createInfer<infer::Infer>("infer",
//   inference_type,
//                                                       input_ids, output);
//   infer->setParam(param[1]);

//   return graph;
// }

// }  // namespace stable_diffusion
// }  // namespace nndeploy
