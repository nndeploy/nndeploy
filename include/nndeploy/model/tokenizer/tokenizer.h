
#ifndef _NNDEPLOY_MODEL_TOKENIZER_TOKENIZER_H_
#define _NNDEPLOY_MODEL_TOKENIZER_TOKENIZER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"

namespace nndeploy {
namespace model {

class NNDEPLOY_CC_API TokenizerPraram : public base::Param {
 public:
  bool is_tokenizer_ = true;
};

class NNDEPLOY_CC_API TokenizerString : public base::Param {
 public:
  std::string str_;
};

/**
 * @brief Tokenizer
 *
 */
class NNDEPLOY_CC_API Tokenizer : public dag::Node {
 public:
  Tokenizer(const std::string &name, dag::Edge *input, dag::Edge *output);
  virtual ~Tokenizer();

  virtual base::Status run() = 0;

  virtual base::Status tokenize(const std::string &str,
                                std::vector<std::string> &tokens) = 0;
  virtual base::Status detokenize(const std::vector<std::string> &tokens,
                                  std::string &str) = 0;

 private:
  std::string dict_path_;
  std::unordered_map<std::string, int> word_dict_;
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_TOKENIZER_TOKENIZER_H_ */
