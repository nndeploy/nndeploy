
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace tokenizer {

std::string tokenizerTypeToString(TokenizerType type) {
  switch (type) {
    case kTokenizerTypeHF:
      return "kTokenizerTypeHF";
    case kTokenizerTypeBPE:
      return "kTokenizerTypeBPE";  
    case kTokenizerTypeSentencePiece:
      return "kTokenizerTypeSentencePiece";
    case kTokenizerTypeRWKVWorld:
      return "kTokenizerTypeRWKVWorld";
    case kTokenizerTypeNotSupport:
      return "kTokenizerTypeNotSupport";
  }
  return "kTokenizerTypeNotSupport";
}

TokenizerType stringToTokenizerType(const std::string &src) {
  if (src == "kTokenizerTypeHF") {
    return kTokenizerTypeHF;
  } else if (src == "kTokenizerTypeBPE") {
    return kTokenizerTypeBPE;
  } else if (src == "kTokenizerTypeSentencePiece") {
    return kTokenizerTypeSentencePiece;
  } else if (src == "kTokenizerTypeRWKVWorld") {
    return kTokenizerTypeRWKVWorld;
  }
  return kTokenizerTypeNotSupport;
}

TokenizerDecode::~TokenizerDecode() {}

TokenizerEncode::~TokenizerEncode() {}

}  // namespace tokenizer
}  // namespace nndeploy
