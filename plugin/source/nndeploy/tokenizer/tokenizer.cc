
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace tokenizer {

Tokenizer::Tokenizer(const std::string &name, dag::Edge *input,
                     dag::Edge *output)
    : dag::Node(name, input, output) {}

Tokenizer::~Tokenizer() {}

}  // namespace tokenizer
}  // namespace nndeploy
