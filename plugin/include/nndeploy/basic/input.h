#ifndef _NNDEPLOY_BASIC_INPUT_H_
#define _NNDEPLOY_BASIC_INPUT_H_

#include "nndeploy/base/half.h"
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

class NNDEPLOY_CC_API InputNum : public dag::Node {
 public:
  InputNum(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::basic::InputNum";
    desc_ = "Input Number Node";
    this->setOutputTypeInfo<double>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeNum);
  }
  InputNum(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::basic::InputNum";
    desc_ = "Input Number Node";
    this->setOutputTypeInfo<double>();
    this->setNodeType(dag::NodeType::kNodeTypeInput);
    this->setIoType(dag::IOType::kIOTypeNum);
  }

  virtual ~InputNum() {}

  virtual base::Status run() {
    auto output_edge = this->getOutput(0);

    if (data_type_ == base::dataTypeOf<float>()) {
      float *num = new float();
      *num = static_cast<float>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<double>()) {
      double *num = new double();
      *num = num_;
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<base::bfp16_t>()) {
      float num = static_cast<float>(num_);
      base::bfp16_t *bfp16 = new base::bfp16_t();
      base::convertFromFloatToBfp16(&num, bfp16, 1);
      output_edge->set(bfp16, false);
    } else if (data_type_ == base::dataTypeOf<half_float::half>()) {
      float num = static_cast<float>(num_);
      half_float::half *half = new half_float::half();
      base::convertFromFloatToFp16(&num, half, 1);
      output_edge->set(half, false);
    } else if (data_type_ == base::dataTypeOf<uint8_t>()) {
      uint8_t *num = new uint8_t();
      *num = static_cast<uint8_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<uint16_t>()) {
      uint16_t *num = new uint16_t();
      *num = static_cast<uint16_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<uint32_t>()) {
      uint32_t *num = new uint32_t();
      *num = static_cast<uint32_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<uint64_t>()) {
      uint64_t *num = new uint64_t();
      *num = static_cast<uint64_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<int8_t>()) {
      int8_t *num = new int8_t();
      *num = static_cast<int8_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<int16_t>()) {
      int16_t *num = new int16_t();
      *num = static_cast<int16_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<int32_t>()) {
      int32_t *num = new int32_t();
      *num = static_cast<int32_t>(num_);
      output_edge->set(num, false);
    } else if (data_type_ == base::dataTypeOf<int64_t>()) {
      int64_t *num = new int64_t();
      *num = static_cast<int64_t>(num_);
      output_edge->set(num, false);
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

 private:
  base::DataType data_type_ = base::dataTypeOf<float>();
  double num_ = 0.0;
};

}  // namespace basic
}  // namespace nndeploy

#endif /* _NNDEPLOY_BASIC_INPUT_H_ */
