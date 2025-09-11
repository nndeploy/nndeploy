#ifndef _NNDEPLOY_BASIC_FILES_H_
#define _NNDEPLOY_BASIC_FILES_H_

#include "nndeploy/base/half.h"
#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace basic {

class NNDEPLOY_CC_API InputCppTextFile : public dag::Node {
 public:
  InputCppTextFile(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::InputCppTextFile";
    desc_ = "Txt File input Node";
    this->setOutputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeText);
  }
  InputCppTextFile(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::InputCppTextFile";
    desc_ = "Txt File input Node";
    this->setOutputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeText);
  }

  virtual ~InputCppTextFile() {}

  virtual base::Status run() {
    auto output_edge = this->getOutput(0);
    std::string content = "";
    std::ifstream file(path_);
    if (file.is_open()) {
      std::string line;
      while (getline(file, line)) {
        content += line + "\n";
      }
      file.close();
    }
    output_edge->set(content);
    return base::Status::Ok();
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    this->addRequiredParam("path_");
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::Status::Ok()) {
      return status;
    }
    json.AddMember("path_", rapidjson::Value(path_.c_str(), allocator),
                   allocator);
    return base::Status::Ok();
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("path_") && json["path_"].IsString()) {
      path_ = json["path_"].GetString();
    }
    return base::Status::Ok();
  }

 private:
  std::string path_ = "";
};

}  // namespace basic
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASIC_INPUT_H_ */
