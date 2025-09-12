#ifndef _NNDEPLOY_BASIC_OUTPUT_H_
#define _NNDEPLOY_BASIC_OUTPUT_H_

#include "nndeploy/base/half.h"
#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace basic {

class NNDEPLOY_CC_API OutputCppStr : public dag::Node {
 public:
  OutputCppStr(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::OutputCppStr";
    desc_ = "Output String Node";
    this->setInputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeString);
  }
  OutputCppStr(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::OutputCppStr";
    desc_ = "Output String Node";
    this->setInputTypeInfo<std::string>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeString);
  }

  virtual ~OutputCppStr() {}

  virtual base::Status run() {
    auto input_edge = this->getInput(0);
    std::string str = *(input_edge->get<std::string>(this));
    str_ = str;
    return base::Status::Ok();
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    this->addRequiredParam("str_");
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::Status::Ok()) {
      return status;
    }
    json.AddMember("str_", rapidjson::Value(str_.c_str(), allocator),
                   allocator);
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

  std::string getOutputValue() const { return str_; }

 private:
  std::string str_ = "";
};

class NNDEPLOY_CC_API OutputCppBool : public dag::Node {
 public:
  OutputCppBool(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::OutputCppBool";
    desc_ = "Output Bool Node";
    this->setInputTypeInfo<bool>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeBool);
  }
  OutputCppBool(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::OutputCppBool";
    desc_ = "Output Bool Node";
    this->setInputTypeInfo<bool>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeBool);
  }

  virtual ~OutputCppBool() {}

  virtual base::Status run() {
    auto input_edge = this->getInput(0);
    flag_ = *(input_edge->get<bool>(this));
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

  bool getOutputValue() const { return flag_; }

 private:
  bool flag_ = false;
};

class NNDEPLOY_CC_API OutputCppNum : public dag::Node {
 public:
  OutputCppNum(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::OutputCppNum";
    desc_ = "Output Number Node";
    this->setInputTypeInfo<double>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeNum);
  }
  OutputCppNum(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::OutputCppNum";
    desc_ = "Output Number Node";
    this->setInputTypeInfo<double>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeNum);
  }

  virtual ~OutputCppNum() {}

  virtual base::Status run() {
    auto input_edge = this->getInput(0);

    if (data_type_ == base::dataTypeOf<float>()) {
      float *num = input_edge->get<float>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<double>()) {
      double *num = input_edge->get<double>(this);
      num_ = *num;
    } else if (data_type_ == base::dataTypeOf<base::bfp16_t>()) {
      base::bfp16_t *bfp16 = input_edge->get<base::bfp16_t>(this);
      float temp;
      base::convertFromBfp16ToFloat(bfp16, &temp, 1);
      num_ = (double)temp;
    } else if (data_type_ == base::dataTypeOf<half_float::half>()) {
      half_float::half *half = input_edge->get<half_float::half>(this);
      float temp;
      base::convertFromFp16ToFloat(half, &temp, 1);
      num_ = (double)temp;
    } else if (data_type_ == base::dataTypeOf<uint8_t>()) {
      uint8_t *num = input_edge->get<uint8_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<uint16_t>()) {
      uint16_t *num = input_edge->get<uint16_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<uint32_t>()) {
      uint32_t *num = input_edge->get<uint32_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<uint64_t>()) {
      uint64_t *num = input_edge->get<uint64_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<int8_t>()) {
      int8_t *num = input_edge->get<int8_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<int16_t>()) {
      int16_t *num = input_edge->get<int16_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<int32_t>()) {
      int32_t *num = input_edge->get<int32_t>(this);
      num_ = (double)(*num);
    } else if (data_type_ == base::dataTypeOf<int64_t>()) {
      int64_t *num = input_edge->get<int64_t>(this);
      num_ = (double)(*num);
    } else {
      NNDEPLOY_LOGE("Unsupported data type: %s.\n", base::dataTypeToString(data_type_).c_str());
      return base::Status(base::kStatusCodeErrorInvalidParam);
    }
    return base::Status::Ok();
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    this->addRequiredParam("num_");
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::Status::Ok()) {
      return status;
    }
    std::string data_type_str = base::dataTypeToString(data_type_);
    json.AddMember("data_type_", rapidjson::Value(data_type_str.c_str(), allocator), allocator);
    json.AddMember("num_", num_, allocator);
    return base::Status::Ok();
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("data_type_") && json["data_type_"].IsString()) {
      data_type_ = base::stringToDataType(json["data_type_"].GetString());
    }
    if (json.HasMember("num_") && json["num_"].IsNumber()) {
      num_ = json["num_"].GetDouble();
    }
    return base::Status::Ok();
  }

  double getOutputValue() const { return num_; }

 private:
  base::DataType data_type_ = base::dataTypeOf<float>();
  double num_ = 0.0;
};

}  // namespace basic
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASIC_OUTPUT_H_ */
