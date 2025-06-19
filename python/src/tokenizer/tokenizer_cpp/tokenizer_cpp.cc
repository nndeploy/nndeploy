
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace tokenizer {

NNDEPLOY_API_PYBIND11_MODULE("tokenizer_cpp", m) {
  py::class_<TokenizerEncodeCpp, TokenizerEncode>(m, "TokenizerEncodeCpp")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &TokenizerEncodeCpp::run)
      .def("init", &TokenizerEncodeCpp::init)
      .def("deinit", &TokenizerEncodeCpp::deinit)
      .def("encode", &TokenizerEncodeCpp::encode)
      .def("encode_batch", &TokenizerEncodeCpp::encodeBatch)
      .def("get_vocab_size", &TokenizerEncodeCpp::getVocabSize)
      .def("token_to_id", &TokenizerEncodeCpp::tokenToId);

  py::class_<TokenizerDecodeCpp, TokenizerDecode>(m, "TokenizerDecodeCpp")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &TokenizerDecodeCpp::run)
      .def("init", &TokenizerDecodeCpp::init)
      .def("deinit", &TokenizerDecodeCpp::deinit)
      .def("decode", &TokenizerDecodeCpp::decode)
      .def("decode_batch", &TokenizerDecodeCpp::decodeBatch)
      .def("get_vocab_size", &TokenizerDecodeCpp::getVocabSize)
      .def("id_to_token", &TokenizerDecodeCpp::idToToken);

}

}
}  // namespace nndeploy
