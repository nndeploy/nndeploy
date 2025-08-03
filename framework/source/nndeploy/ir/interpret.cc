#include "nndeploy/ir/interpret.h"

#include "nndeploy/base/status.h"
#include "safetensors.hh"

namespace nndeploy {
namespace ir {

// Interpret::Interpret() { model_desc_ = new ModelDesc(); }

Interpret::Interpret(ModelDesc *model_desc, bool is_external) {
  if (nullptr != model_desc) {
    model_desc_ = model_desc;
    is_external_ = is_external;
  } else {
    model_desc_ = new ModelDesc();
    is_external_ = false;
  }
}

Interpret::~Interpret() {
  if (model_desc_ != nullptr) {
    delete model_desc_;
  }
}

base::Status Interpret::dump(std::ostream &oss) {
  std::string structure_str;
  base::Status status = model_desc_->serializeStructureToJsonStr(structure_str);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("model_desc_->serializeStructureToJsonStr failed!\n");
    return status;
  }
  oss.write(structure_str.c_str(), structure_str.size());
  return base::kStatusCodeOk;
}

base::Status Interpret::saveModel(
    std::string &structure_str,
    std::shared_ptr<safetensors::safetensors_t> st_ptr) {
  base::Status status = model_desc_->serializeStructureToJsonStr(structure_str);
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
    std::string structure_str;
    base::Status status =
        model_desc_->serializeStructureToJsonStr(structure_str);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("model_desc_->serializeStructureToJson failed!\n");
      return status;
    }
    structure_stream.write(structure_str.c_str(), structure_str.size());
    structure_stream.close();
  }

  // 检查weight_file_path，确保使用'.safetensors'作为权重文件的后缀
  if (!weight_file_path.empty()) {
    std::string path = weight_file_path;
    const std::string extension = ".safetensors";
    size_t pos = weight_file_path.find_last_of('.');
    if (pos == std::string::npos || weight_file_path.substr(pos) != extension) {
      path = weight_file_path + extension;
    }
    std::shared_ptr<safetensors::safetensors_t> st_ptr(
        new safetensors::safetensors_t());

    base::Status status = model_desc_->serializeWeightsToSafetensors(st_ptr);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("model_desc_->serializeWeightsToBinary failed!\n");
      return status;
    }

    std::string warn, err;
    bool ret = safetensors::save_to_file((*st_ptr), path, &warn, &err);
    if (warn.size()) {
      NNDEPLOY_LOGI("WARN: %s\n", warn.c_str());
    }
    if (!ret) {
      NNDEPLOY_LOGE("Failed to load: %s\nERR: %s", path.c_str(), err.c_str());
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

Interpret *createInterpret(base::ModelType type, ir::ModelDesc *model_desc,
                           bool is_external) {
  Interpret *temp = nullptr;
  auto &creater_map = getGlobalInterpretCreatorMap();
  if (creater_map.count(type) > 0) {
    // NNDEPLOY_LOGE("createInterpret: %d\n", type);
    temp = creater_map[type]->createInterpret(type, model_desc, is_external);
    // NNDEPLOY_LOGE("createInterpret: %p\n", temp);
  }
  return temp;
}

std::shared_ptr<Interpret> createInterpretSharedPtr(base::ModelType type,
                                                    ir::ModelDesc *model_desc,
                                                    bool is_external) {
  std::shared_ptr<Interpret> temp = nullptr;
  auto &creater_map = getGlobalInterpretCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createInterpretSharedPtr(type, model_desc,
                                                       is_external);
  }
  return temp;
}

}  // namespace ir
}  // namespace nndeploy