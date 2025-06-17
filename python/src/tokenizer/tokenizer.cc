#include "nndeploy/tokenizer/tokenizer.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace tokenizer {

class PyTokenizerEncode : public TokenizerEncode {
 public:
  using TokenizerEncode::TokenizerEncode;

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status, TokenizerEncode, run);
  }
};

class PyTokenizerDecode : public TokenizerDecode {
 public:
  using TokenizerDecode::TokenizerDecode;

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status, TokenizerDecode, run);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("tokenizer", m) {
  py::enum_<TokenizerType>(m, "TokenizerType")
      .value("kTokenizerTypeHF", TokenizerType::kTokenizerTypeHF)
      .value("kTokenizerTypeBPE", TokenizerType::kTokenizerTypeBPE)
      .value("kTokenizerTypeSentencePiece",
             TokenizerType::kTokenizerTypeSentencePiece)
      .value("kTokenizerTypeRWKVWorld", TokenizerType::kTokenizerTypeRWKVWorld)
      .value("kTokenizerTypeNotSupport",
             TokenizerType::kTokenizerTypeNotSupport);

  m.def("tokenizer_type_to_string", &tokenizerTypeToString);
  m.def("string_to_tokenizer_type", &stringToTokenizerType);

  py::class_<TokenizerPraram, base::Param, std::shared_ptr<TokenizerPraram>>(
      m, "TokenizerPraram")
      .def(py::init<>())
      .def_readwrite("is_path_", &TokenizerPraram::is_path_)
      .def_readwrite("tokenizer_type_", &TokenizerPraram::tokenizer_type_)
      .def_readwrite("json_blob_", &TokenizerPraram::json_blob_)
      .def_readwrite("model_blob_", &TokenizerPraram::model_blob_)
      .def_readwrite("vocab_blob_", &TokenizerPraram::vocab_blob_)
      .def_readwrite("merges_blob_", &TokenizerPraram::merges_blob_)
      .def_readwrite("added_tokens_", &TokenizerPraram::added_tokens_)
      .def_readwrite("max_length_", &TokenizerPraram::max_length_);

  py::class_<TokenizerText, base::Param, std::shared_ptr<TokenizerText>>(
      m, "TokenizerText")
      .def(py::init<>())
      .def_readwrite("texts_", &TokenizerText::texts_);

  py::class_<TokenizerIds, base::Param, std::shared_ptr<TokenizerIds>>(
      m, "TokenizerIds")
      .def(py::init<>())
      .def_readwrite("ids_", &TokenizerIds::ids_);

  py::class_<TokenizerEncode, PyTokenizerEncode, dag::Node>(
      m, "TokenizerEncode")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &TokenizerEncode::run);

  py::class_<TokenizerDecode, PyTokenizerDecode, dag::Node>(
      m, "TokenizerDecode")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &TokenizerDecode::run);
      
}

}  // namespace tokenizer
}  // namespace nndeploy