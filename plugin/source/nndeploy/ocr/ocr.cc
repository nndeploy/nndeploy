

#include "nndeploy/ocr/ocr.h"

#include "nndeploy/ocr/utils.h"

namespace nndeploy {
namespace ocr {

base::Status RotateCropImage::run() {
  cv::Mat *src = inputs_[1]->getCvMat(this);
  auto edge_ptr = inputs_[0]->get<OCRResult>(this);
  auto detector_result = static_cast<OCRResult *>(edge_ptr);

  // 使用 detector_results
  int origin_w = int(src->cols);
  int origin_h = int(src->rows);
  if (detector_result->boxes_.size() == 0) {
    return base::kStatusCodeOk;
  } else {
    detector_result->image_list_.resize(detector_result->boxes_.size());
    std::vector<std::vector<std::vector<int>>> boxes_recovered;

    for (auto &arr : detector_result->boxes_) {
      std::vector<std::vector<int>> one_box;
      for (int i = 0; i < 8; i += 2) {
        one_box.push_back({arr[i], arr[i + 1]});  // 每两个数就是一个点 (x,y)
      }
      boxes_recovered.push_back(one_box);
    }

    boxes_recovered = util_post_processor_.FilterTagDetRes(
        boxes_recovered,
        {origin_w, origin_h, detector_result->detector_resized_w,
         detector_result->detector_resized_h});
    detector_result->image_list_.clear();
    for (int i = 0; i < boxes_recovered.size(); i++) {
      std::array<int, 8> new_box;
      int k = 0;
      for (auto &vec : boxes_recovered[i]) {
        for (auto &e : vec) {
          new_box[k++] = e;
        }
      }

      detector_result->image_list_.emplace_back(
          ocr::GetRotateCropImage(*src, new_box));
    }
  }

  outputs_[0]->set(detector_result, true);

  return base::kStatusCodeOk;
}

base::Status RotateImage180::run() {
  auto classifier_edge_ptr = inputs_[0]->get<OCRResult>(this);
  auto ocr_classifier_result = static_cast<OCRResult *>(classifier_edge_ptr);

  auto images_edge_ptr = inputs_[1]->get<OCRResult>(this);
  auto crop_images_result = static_cast<OCRResult *>(images_edge_ptr);

  for (int i = 0; i < crop_images_result->image_list_.size(); i++) {
    if (ocr_classifier_result->classifier_result[i] == 1) {
      cv::rotate(crop_images_result->image_list_[i],
                 crop_images_result->image_list_[i], 1);
    }
  }

  outputs_[0]->set(crop_images_result, true);

  return base::kStatusCodeOk;
}

base::Status PrintOcrNodeParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  this->addRequiredParam("path_");
  base::Status status = base::Param::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("PrintOcrNodeParam::serialize failed\n");
    return status;
  }
  json.AddMember("path_", rapidjson::Value(path_.c_str(), allocator),
                 allocator);
  return base::kStatusCodeOk;
}

base::Status PrintOcrNodeParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("path_") && json["path_"].IsString()) {
    path_ = json["path_"].GetString();
  }
  return base::kStatusCodeOk;
}

base::Status PrintOcrNode::run() {
  PrintOcrNodeParam *tmp_param =
      dynamic_cast<PrintOcrNodeParam *>(param_.get());

  auto edge_ptr = inputs_[0]->get<OCRResult>(this);
  auto result = static_cast<OCRResult *>(edge_ptr);
  if (result == nullptr) {
    return base::kStatusCodeErrorInvalidValue;
  }

  // 把所有 text_ 拼接成一个字符串
  std::ofstream file(tmp_param->path_);
  if (file.is_open()) {
    for (size_t i = 0; i < result->text_.size(); i++) {
      file << result->text_[i];
      file << "||";
      file << std::to_string(result->rec_scores_[i]);
      file << "\n";
    }
    file.close();
  } else {
    NNDEPLOY_LOGE("Unable to open text file for writing");
    return base::kStatusCodeErrorIO;
  }
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::ocr::RotateImage180", RotateImage180);
REGISTER_NODE("nndeploy::ocr::RotateCropImage", RotateCropImage);
REGISTER_NODE("nndeploy::ocr::PrintOcrNode", PrintOcrNode);
}  // namespace ocr
}  // namespace nndeploy