#ifndef _NNDEPLOY_BASIC_INPUT_H_
#define _NNDEPLOY_BASIC_INPUT_H_

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace basic {

class NNDEPLOY_CC_API InputString : public dag::Node {
 public:
  InputString(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::InputString";
    desc_ = "Input String Node";
    this->setOutputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeString);
  }
  InputString(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::InputString";
    desc_ = "Input String Node";
    this->setOutputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeString);
  }

  virtual ~InputString() {}

  virtual base::Status run() {
    auto output_edge = this->getOutput(0);
    output_edge->set(str_);
    return base::Status::Ok();
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    this->addRequiredParam("str_");
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::Status::Ok()) {
      return status;
    }
    json.AddMember("str_", rapidjson::Value(str_.c_str(), allocator), allocator);
    return base::Status::Ok();
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("str_") && json["str_"].IsString()) {
      str_ = json["str_"].GetString();
    }
    return base::Status::Ok();
  }

 private:
  std::string str_ = "";
};

class NNDEPLOY_CC_API InputBool : public dag::Node {
 public:
  InputBool(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::InputBool";
    desc_ = "Input Bool Node";
    this->setOutputTypeInfo<bool>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeBool);
  }
  InputBool(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::InputBool";
    desc_ = "Input Bool Node";
    this->setOutputTypeInfo<bool>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeBool);
  }

  virtual ~InputBool() {}

  virtual base::Status run() {
    auto output_edge = this->getOutput(0);
    output_edge->set(flag_);
    return base::Status::Ok();
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    this->addRequiredParam("flag_");
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::Status::Ok()) {
      return status;
    }
    json.AddMember("flag_", flag_, allocator);
    return base::Status::Ok();
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("flag_") && json["flag_"].IsBool()) {
      flag_ = json["flag_"].GetBool();
    }
    return base::Status::Ok();
  }

 private:
  bool flag_ = false;
};

}  // namespace basic
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASIC_INPUT_H_ */
