#include <tokenizers_cpp.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/model/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

using tokenizers::Tokenizer;

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

void PrintEncodeResult(const std::vector<int>& ids) {
  std::cout << "tokens=[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

void TestTokenizer(std::unique_ptr<Tokenizer> tok, bool print_vocab = false,
                   bool check_id_back = true) {
  // Check #1. Encode and Decode
  std::string prompt = "What is the  capital of Canada?";
  std::vector<int> ids = tok->Encode(prompt);
  std::string decoded_prompt = tok->Decode(ids);
  PrintEncodeResult(ids);
  std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
  assert(decoded_prompt == prompt);

  // Check #2. IdToToken and TokenToId
  std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 32, 33, 34, 130, 131, 1000};
  for (auto id : ids_to_test) {
    auto token = tok->IdToToken(id);
    auto id_new = tok->TokenToId(token);
    std::cout << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new
              << std::endl;
    if (check_id_back) {
      assert(id == id_new);
    }
  }

  // Check #3. GetVocabSize
  auto vocab_size = tok->GetVocabSize();
  std::cout << "vocab_size=" << vocab_size << std::endl;

  std::cout << std::endl;
}

// Sentencepiece tokenizer
// - ./tokenizer.model
void SentencePieceTokenizerExample() {
  std::cout << "Tokenizer: SentencePiece" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Read blob from file.
  auto blob = LoadBytesFromFile("./tokenizer.model");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobSentencePiece(blob);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "Load time: " << duration << " ms" << std::endl;

  TestTokenizer(std::move(tok), false, true);
}

// HF tokenizer
// - ./tokenizer.json
void HuggingFaceTokenizerExample() {
  std::cout << "Tokenizer: Huggingface" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Read blob from file.
  auto blob = LoadBytesFromFile("./tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(blob);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "Load time: " << duration << " ms" << std::endl;

  TestTokenizer(std::move(tok), false, true);
}

// RWKV world tokenizer
// - ./tokenizer_model
void RWKVWorldTokenizerExample() {
  std::cout << "Tokenizer: RWKVWorld" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  auto tok = Tokenizer::FromBlobRWKVWorld("./tokenizer_model");

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "Load time: " << duration << " ms" << std::endl;

  // We cannot check id back for RWKVWorldTokenizer yet.
  TestTokenizer(std::move(tok), false, false);
}

int main(int argc, char* argv[]) {
  SentencePieceTokenizerExample();
  HuggingFaceTokenizerExample();
  RWKVWorldTokenizerExample();

  nndeploy::dag::Edge* input_edge = new nndeploy::dag::Edge("input_edge");
  nndeploy::dag::Edge* output_edge = new nndeploy::dag::Edge("output_edge");
  NNDEPLOY_LOGE("test tokenizer_cpp\n");

  nndeploy::dag::Graph graph("graph", input_edge, output_edge);

  nndeploy::model::TokenizerCpp* tokenizer_cpp =
      (nndeploy::model::TokenizerCpp*)
          graph.createNode<nndeploy::model::TokenizerCpp>("name", input_edge,
                                                          output_edge);
  NNDEPLOY_LOGE("test tokenizer_cpp\n");
  nndeploy::model::TokenizerPraram* tp =
      (nndeploy::model::TokenizerPraram*)tokenizer_cpp->getParam();
  if (tp == nullptr) {
    NNDEPLOY_LOGE("test tokenizer_cpp\n");
    return -1;
  }
  NNDEPLOY_LOGE("test tokenizer_cpp\n");
  NNDEPLOY_LOGE("%p\n", tp);
  tp->is_encode_ = true;

  tp->tokenizer_type_ =
      nndeploy::model::TokenizerType::kTokenizerTypeSentencePiece;
  NNDEPLOY_LOGE("test tokenizer_cpp\n");
  auto blob = LoadBytesFromFile("./tokenizer.model");
  tp->model_blob_ = blob;
  NNDEPLOY_LOGE("test tokenizer_cpp\n");
  // tokenizer_cpp->setParam((nndeploy::base::Param*)tp);
  NNDEPLOY_LOGE("test tokenizer_cpp\n");

  graph.init();
  NNDEPLOY_LOGE("test tokenizer_cpp\n");

  std::string prompt = "What is the  capital of Canada?";
  nndeploy::model::TokenizerText tt;
  tt.texts_.push_back(prompt);
  NNDEPLOY_LOGE("test tokenizer_cpp\n");
  input_edge->set((nndeploy::base::Param*)(&tt), 0);
  NNDEPLOY_LOGE("test tokenizer_cpp\n");

  auto vocab_size = tokenizer_cpp->getVocabSize();
  std::cout << "vocab_size=" << vocab_size << std::endl;
  NNDEPLOY_LOGE("test tokenizer_cpp\n");

  graph.run();

  nndeploy::model::TokenizerIds* ti =
      (nndeploy::model::TokenizerIds*)(output_edge->getGraphOutputParam());

  std::vector<int> ids = ti->ids_[0];
  PrintEncodeResult(ids);

  return 0;
}
