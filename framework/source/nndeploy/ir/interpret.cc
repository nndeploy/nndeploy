#include "nndeploy/ir/interpret.h"

namespace nndeploy {
namespace ir {

Interpret::Interpret() { model_desc_ = new ModelDesc(); }
Interpret::~Interpret() {
  if (model_desc_ != nullptr) {
    delete model_desc_;
  }
}

base::Status Interpret::dump(std::ostream &oss) {
  return model_desc_->serializeStructureToText(oss);
}

base::Status Interpret::saveModel(std::ostream &structure_stream,
                                  std::ostream &weight_stream) {
  base::Status status = model_desc_->serializeStructureToText(structure_stream);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("model_desc_->serializeStructureToText failed!\n");
    return status;
  }
  status = model_desc_->serializeWeightsToBinary(weight_stream);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("model_desc_->serializeWeightsToBinary failed!\n");
    return status;
  }
  return status;
}

base::Status Interpret::saveModelToFile(const std::string &structure_file_path,
                                        const std::string &weight_file_path) {
  // 打开结构文件输出流，覆盖已存在文件
  std::ofstream structure_stream(
      structure_file_path,
      std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
  if (!structure_stream.is_open()) {
    NNDEPLOY_LOGE("Failed to open structure file: %s\n",
                  structure_file_path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  // 打开权重文件输出流，覆盖已存在文件
  std::ofstream weight_stream(weight_file_path, std::ofstream::out |
                                                    std::ofstream::trunc |
                                                    std::ofstream::binary);
  if (!weight_stream.is_open()) {
    NNDEPLOY_LOGE("Failed to open weight file: %s\n", weight_file_path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  // 保存模型结构和权重
  base::Status status = saveModel(structure_stream, weight_stream);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Failed to save model to files: %s, %s\n",
                  structure_file_path.c_str(), weight_file_path.c_str());
  }

  return status;
}

ModelDesc *Interpret::getModelDesc() { return model_desc_; }

}  // namespace ir
}  // namespace nndeploy