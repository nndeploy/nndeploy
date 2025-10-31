#ifndef _NNDEPLOY_BASIC_END_H_
#define _NNDEPLOY_BASIC_END_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace basic {

class NNDEPLOY_CC_API End : public dag::Node {
 public:
  End(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::End";
    desc_ = "End Node";
    this->setInputTypeInfo<base::Any>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeAny);
  }
  End(const std::string &name, std::vector<dag::Edge *> inputs,
      std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::End";
    desc_ = "End Node";
    this->setInputTypeInfo<base::Any>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeAny);
  }

  virtual ~End() {}

  virtual base::Status run();
};

}  // namespace basic
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASIC_END_H_ */
