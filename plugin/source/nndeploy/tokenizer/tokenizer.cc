
#include "nndeploy/model/tokenizer/tokenizer.h"

namespace nndeploy {
namespace model {

Tokenizer::Tokenizer(const std::string &name, dag::Edge *input,
                     dag::Edge *output)
    : dag::Node(name, input, output) {}

Tokenizer::~Tokenizer() {}

}  // namespace model
}  // namespace nndeploy
