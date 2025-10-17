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
    std::string *content = new std::string();
    std::ifstream file(path_);
    if (file.is_open()) {
      std::string line;
      while (getline(file, line)) {
        *content += line + "\n";
      }
      file.close();
    }
    output_edge->set(content, false);
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

class NNDEPLOY_CC_API InputCppBinaryFile : public dag::Node {
 public:
  InputCppBinaryFile(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::InputCppBinaryFile";
    desc_ = "Txt File input Node";
    this->setOutputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeBinary);
  }
  InputCppBinaryFile(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::InputCppBinaryFile";
    desc_ = "Txt File input Node";
    this->setOutputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeBinary);
  }

  virtual ~InputCppBinaryFile() {}

  virtual base::Status run() {
    auto output_edge = this->getOutput(0);
    std::string *content = new std::string();
    std::ifstream file(path_, std::ios::binary);
    if (file.is_open()) {
      // Get file size
      file.seekg(0, std::ios::end);
      std::streamsize size = file.tellg();
      file.seekg(0, std::ios::beg);
      
      // Read binary data
      content->resize(size);
      if (!file.read(&(*content)[0], size)) {
        file.close();
        delete content;
        NNDEPLOY_LOGE("Failed to read binary file");
        return base::kStatusCodeErrorIO;
      }
      file.close();
    } else {
      delete content;
      NNDEPLOY_LOGE("Unable to open binary file");
      return base::kStatusCodeErrorIO;;
    }
    output_edge->set(content, false);
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


class NNDEPLOY_CC_API OutputCppTextFile : public dag::Node {
 public:
  OutputCppTextFile(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::OutputCppTextFile";
    desc_ = "Txt File output Node";
    this->setInputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeText);
  }
  OutputCppTextFile(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::OutputCppTextFile";
    desc_ = "Txt File output Node";
    this->setInputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeText);
  }

  virtual ~OutputCppTextFile() {}

  virtual base::Status run() {
    auto input_edge = this->getInput(0);
    std::string *content = input_edge->get<std::string>(this);
    if (content == nullptr) {
      NNDEPLOY_LOGE("Input content is null");
      return base::kStatusCodeErrorNullParam;
    }
    
    std::ofstream file(path_);
    if (file.is_open()) {
      file << *content;
      file.close();
    } else {
      NNDEPLOY_LOGE("Unable to open text file for writing");
      return base::kStatusCodeErrorIO;
    }
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

class NNDEPLOY_CC_API OutputCppBinaryFile : public dag::Node {
 public:
  OutputCppBinaryFile(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::OutputCppBinaryFile";
    desc_ = "Binary File output Node";
    this->setInputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeBinary);
  }
  OutputCppBinaryFile(const std::string &name, std::vector<dag::Edge *> inputs,
                      std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::OutputCppBinaryFile";
    desc_ = "Binary File output Node";
    this->setInputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeBinary);
  }

  virtual ~OutputCppBinaryFile() {}

  virtual base::Status run() {
    auto input_edge = this->getInput(0);
    std::string *content = input_edge->get<std::string>(this);
    if (content == nullptr) {
      NNDEPLOY_LOGE("Input content is null");
      return base::kStatusCodeErrorNullParam;
    }
    
    std::ofstream file(path_, std::ios::binary);
    if (file.is_open()) {
      file.write(content->data(), content->size());
      file.close();
    } else {
      NNDEPLOY_LOGE("Unable to open binary file for writing");
      return base::kStatusCodeErrorIO;
    }
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
