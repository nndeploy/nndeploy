#include "nndeploy/ir/interpret.h"
#include "nndeploy/base/status.h"
#include "safetensors.hh"

namespace nndeploy {
namespace ir {

Interpret::Interpret() { model_desc_ = new ModelDesc(); }
Interpret::~Interpret() {
  if (model_desc_ != nullptr) {
    delete model_desc_;
  }
}

base::Status Interpret::dump(std::ostream &oss) {
  return model_desc_->serializeStructureToJson(oss);
}

base::Status Interpret::saveModel(
    std::ostream &structure_stream, std::shared_ptr<safetensors::safetensors_t> st_ptr) {
  base::Status status = model_desc_->serializeStructureToJson(structure_stream);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("model_desc_->serializeStructureToJson failed!\n");
    return status;
  }
  status = model_desc_->serializeWeightsToSafetensors(st_ptr);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("model_desc_->serializeWeightsToBinary failed!\n");
    return status;
  }
  return status;
}

base::Status Interpret::saveModelToFile(const std::string &structure_file_path,
                                        const std::string &weight_file_path) {
  // 打开结构文件输出流，覆盖已存在文件
  if (!structure_file_path.empty()) {
    std::ofstream structure_stream(
      structure_file_path,
      std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
    if (!structure_stream.is_open()) {
      NNDEPLOY_LOGE("Failed to open structure file: %s\n",
                  structure_file_path.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    base::Status status = model_desc_->serializeStructureToJson(structure_stream);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("model_desc_->serializeStructureToJson failed!\n");
      return status;
    }
  }

  // 检查weight_file_path，确保使用'.safetensors'作为权重文件的后缀
  if (!weight_file_path.empty()) {
    std::string path = weight_file_path;
    const std::string extension = ".safetensors";
    size_t pos = weight_file_path.find_last_of('.');
    if (pos == std::string::npos || weight_file_path.substr(pos) != extension) {
      path = weight_file_path + extension;
    }
    std::shared_ptr<safetensors::safetensors_t> st_ptr(new safetensors::safetensors_t());

    base::Status status = model_desc_->serializeWeightsToSafetensors(st_ptr);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("model_desc_->serializeWeightsToBinary failed!\n");
      return status;
    }

    std::string warn, err;
    bool ret =
        safetensors::save_to_file((*st_ptr), path, &warn, &err);
    if (warn.size()) {
      NNDEPLOY_LOGI("WARN: %s\n", warn.c_str());
    }
    if (!ret) {
      NNDEPLOY_LOGE("Failed to load: %s\nERR: %s", path.c_str(),
                    err.c_str());
      return base::kStatusCodeErrorIO;
    }
  }

  return base::kStatusCodeOk;
}

ModelDesc *Interpret::getModelDesc() { return model_desc_; }

std::map<base::ModelType, std::shared_ptr<InterpretCreator>> &
getGlobalInterpretCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::ModelType, std::shared_ptr<InterpretCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::ModelType, std::shared_ptr<InterpretCreator>>);
  });
  return *creators;
}

Interpret *createInterpret(base::ModelType type) {
  Interpret *temp = nullptr;
  auto &creater_map = getGlobalInterpretCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInterpret(type);
  }
  return temp;
}

}  // namespace ir
}  // namespace nndeploy