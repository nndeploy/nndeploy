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


base::Status Param::setRequiredParams(const std::vector<std::string> &required_params) {
  required_params_ = required_params;
  return base::kStatusCodeOk;
}
base::Status Param::addRequiredParam(const std::string &required_param) {
  required_params_.emplace_back(required_param);
  return base::kStatusCodeOk;
}
base::Status Param::removeRequiredParam(const std::string &required_param) {
  auto it = std::find(required_params_.begin(), required_params_.end(), required_param);
  if (it != required_params_.end()) {
    required_params_.erase(it);
  }
  return base::kStatusCodeOk;
}
base::Status Param::clearRequiredParams() {
  required_params_.clear();
  return base::kStatusCodeOk;
}
std::vector<std::string> Param::getRequiredParams() {
  return required_params_;
}

base::Status Param::setUiParams(const std::vector<std::string> &ui_params) {
  ui_params_ = ui_params;
  return base::kStatusCodeOk;
}
base::Status Param::addUiParam(const std::string &ui_param) {
  ui_params_.emplace_back(ui_param);
  return base::kStatusCodeOk;
}
base::Status Param::removeUiParam(const std::string &ui_param) {
  auto it = std::find(ui_params_.begin(), ui_params_.end(), ui_param);
  if (it != ui_params_.end()) {
    ui_params_.erase(it);
  }
  return base::kStatusCodeOk;
}
base::Status Param::clearUiParams() {
  ui_params_.clear();
  return base::kStatusCodeOk;
}
std::vector<std::string> Param::getUiParams() {
  return ui_params_;
}

base::Status Param::setIoParams(const std::vector<std::string> &io_params) {
  io_params_ = io_params;
  return base::kStatusCodeOk;
}
base::Status Param::addIoParam(const std::string &io_param) {
  io_params_.emplace_back(io_param);
  return base::kStatusCodeOk;
}
base::Status Param::removeIoParam(const std::string &io_param) {
  auto it = std::find(io_params_.begin(), io_params_.end(), io_param);
  if (it != io_params_.end()) {
    io_params_.erase(it);
  }
  return base::kStatusCodeOk;
}
base::Status Param::clearIoParams() {
  io_params_.clear();
  return base::kStatusCodeOk;
}
std::vector<std::string> Param::getIoParams() {
  return io_params_;
}

base::Status Param::setDropdownParams(const std::map<std::string, std::vector<std::string>> &dropdown_params) {
  dropdown_params_ = dropdown_params;
  return base::kStatusCodeOk;
}
base::Status Param::addDropdownParam(const std::string &dropdown_param, const std::vector<std::string> &dropdown_values) {
  dropdown_params_[dropdown_param] = dropdown_values;
  return base::kStatusCodeOk;
}
base::Status Param::removeDropdownParam(const std::string &dropdown_param) {
  auto it = dropdown_params_.find(dropdown_param);
  if (it != dropdown_params_.end()) {
    dropdown_params_.erase(it);
  }
  return base::kStatusCodeOk;
}
base::Status Param::clearDropdownParams() {
  dropdown_params_.clear();
  return base::kStatusCodeOk;
}
std::map<std::string, std::vector<std::string>> Param::getDropdownParams() {
  return dropdown_params_;
}


// 序列化：数据结构->[rapidjson::Value\stream\path\string]
base::Status Param::serialize(rapidjson::Value &json,
                              rapidjson::Document::AllocatorType &allocator) {
  rapidjson::Value required_params(rapidjson::kArrayType);
  for (auto &required_param : required_params_) {
    required_params.PushBack(rapidjson::Value(required_param.c_str(), allocator),
                             allocator);
  }
  json.AddMember("required_params_", required_params, allocator);
  rapidjson::Value ui_params(rapidjson::kArrayType);
  for (auto &ui_param : ui_params_) {
    ui_params.PushBack(rapidjson::Value(ui_param.c_str(), allocator), allocator);
  }
  json.AddMember("ui_params_", ui_params, allocator);
  rapidjson::Value io_params(rapidjson::kArrayType);
  for (auto &io_param : io_params_) {
    io_params.PushBack(rapidjson::Value(io_param.c_str(), allocator), allocator);
  }
  json.AddMember("io_params_", io_params, allocator);
  rapidjson::Value dropdown_params(rapidjson::kObjectType);
  for (auto &dropdown_param : dropdown_params_) {
    rapidjson::Value dropdown_values(rapidjson::kArrayType);
    for (auto &dropdown_value : dropdown_param.second) {
      dropdown_values.PushBack(rapidjson::Value(dropdown_value.c_str(), allocator), allocator);
    }
    dropdown_params.AddMember(rapidjson::Value(dropdown_param.first.c_str(), allocator), dropdown_values, allocator);
  }
  json.AddMember("dropdown_params_", dropdown_params, allocator);
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
  // if (json.HasMember("required_params_") && json["required_params_"].IsArray()) {
  //   required_params_.clear();
  //   auto &required_params_array = json["required_params_"];
  //   for (int i = 0; i < required_params_array.Size(); i++) {
  //     if (required_params_array[i].IsString()) {
  //       required_params_.emplace_back(required_params_array[i].GetString());
  //     }
  //   }
  // }
  if (json.HasMember("ui_params_") && json["ui_params_"].IsArray()) {
    ui_params_.clear();
    auto &ui_params_array = json["ui_params_"];
    for (int i = 0; i < ui_params_array.Size(); i++) {
      if (ui_params_array[i].IsString()) {
        ui_params_.emplace_back(ui_params_array[i].GetString());
      }
    }
  }
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