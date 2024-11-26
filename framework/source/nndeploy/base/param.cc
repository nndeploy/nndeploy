#include "nndeploy/base/param.h"

namespace nndeploy {
namespace base {

Param::Param() {}

Param::~Param() {}

base::Status Param::set(const std::string &key, base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status Param::get(const std::string &key, base::Any &any) {
  return base::kStatusCodeOk;
}

// 序列化：数据结构->[rapidjson::Value\stream\path\string]
base::Status Param::serialize(rapidjson::Value &json,
                              rapidjson::Document::AllocatorType &allocator) {
  return base::kStatusCodeOk;
}
base::Status Param::serialize(std::ostream &stream) {
  rapidjson::Document doc;
  rapidjson::Value json(rapidjson::kObjectType);
  base::Status status = this->serialize(json, doc.GetAllocator());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize to json failed\n");
    return status;
  }
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  json.Accept(writer);
  stream << buffer.GetString();
  return base::kStatusCodeOk;
}
base::Status Param::serialize(std::string &content, bool is_file) {
  if (is_file) {
    std::ofstream ofs(content);
    if (!ofs.is_open()) {
      NNDEPLOY_LOGE("open file %s failed\n", content.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    base::Status status = this->serialize(ofs);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("serialize to json failed\n");
      return status;
    }
    ofs.close();
    return status;
  } else {
    rapidjson::Document doc;
    rapidjson::Value json(rapidjson::kObjectType);
    base::Status status = this->serialize(json, doc.GetAllocator());
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("serialize to json failed\n");
      return status;
    }
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    content = buffer.GetString();
    return base::kStatusCodeOk;
  }
}
// 反序列化：[rapidjson::Value\stream\path\string]->数据结构
base::Status Param::deserialize(rapidjson::Value &json) {
  return base::kStatusCodeOk;
}
base::Status Param::deserialize(std::istream &stream) {
  std::string json_str;
  std::string line;
  while (std::getline(stream, line)) {
    json_str += line;
  }
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;
  return this->deserialize(json);
}
base::Status Param::deserialize(const std::string &content, bool is_file) {
  if (is_file) {
    std::ifstream ifs(content);
    if (!ifs.is_open()) {
      NNDEPLOY_LOGE("open file %s failed\n", content.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    base::Status status = this->deserialize(ifs);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("deserialize from file %s failed\n", content.c_str());
      return status;
    }
    ifs.close();
    return status;
  } else {
    rapidjson::Document document;
    if (document.Parse(content.c_str()).HasParseError()) {
      NNDEPLOY_LOGE("parse json string failed\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    rapidjson::Value &json = document;
    return this->deserialize(json);
  }
}

}  // namespace base
}  // namespace nndeploy