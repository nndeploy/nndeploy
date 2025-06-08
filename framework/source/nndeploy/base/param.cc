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
std::string Param::serialize() {
  std::string json_str;
  rapidjson::Document doc;
  rapidjson::Value json(rapidjson::kObjectType);
  base::Status status = this->serialize(json, doc.GetAllocator());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize to json failed\n");
    return json_str;
  }
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json.Accept(writer);
  json_str = buffer.GetString();
  if (json_str.empty()) {
    NNDEPLOY_LOGI("serialize to json failed\n");
  }
  return json_str;
}
base::Status Param::saveFile(const std::string &path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  std::string json_str = this->serialize();
  std::string beautify_json_str = base::prettyJsonStr(json_str);
  ofs.write(beautify_json_str.c_str(), beautify_json_str.size());
  ofs.close();
  return base::kStatusCodeOk;
}
// 反序列化：[rapidjson::Value\stream\path\string]->数据结构
base::Status Param::deserialize(rapidjson::Value &json) {
  return base::kStatusCodeOk;
}
base::Status Param::deserialize(const std::string &json_str) {
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;
  return this->deserialize(json);
}
base::Status Param::loadFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  std::string json_str;
  std::string line;
  while (std::getline(ifs, line)) {
    json_str += line;
  }
  base::Status status = this->deserialize(json_str);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize from file %s failed\n", path.c_str());
    return status;
  }
  ifs.close();
  return status;
}

std::string removeJsonBrackets(const std::string &json_str) {
  std::string result = json_str;
  // 移除开头的 {
  size_t start = result.find_first_of('{');
  if (start != std::string::npos) {
    result = result.substr(start + 1);
  }
  
  // 移除结尾的 }
  size_t end = result.find_last_of('}'); 
  if (end != std::string::npos) {
    result = result.substr(0, end);
  }

  // 在结尾添加逗号
  result += ",";

  return result;
}

std::string prettyJsonStr(const std::string &json_str) {
  rapidjson::Document doc;
  doc.Parse(json_str.c_str());
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  return buffer.GetString();
}

}  // namespace base
}  // namespace nndeploy