#include "nndeploy/ocr/recognizer.h"
#include "nndeploy/ocr/result.h"
#include "nndeploy/preprocess/opencv_convert.h"
#include "nndeploy/preprocess/opencv_util.h"
#include "nndeploy/preprocess/util.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

namespace nndeploy {
namespace ocr {

void filterTexts(const std::vector<std::string>& texts,
                 const std::vector<float>& rec_scores, float rec_thresh_,
                 std::vector<std::string>& filtered_texts,
                 std::vector<float>& filtered_scores) {
  filtered_texts.clear();
  filtered_scores.clear();

  for (size_t i = 0; i < rec_scores.size(); ++i) {
    if (rec_scores[i] > rec_thresh_) {
      filtered_texts.push_back(texts[i]);
      filtered_scores.push_back(rec_scores[i]);
    }
  }
}

base::Status RecognizerPostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  this->addRequiredParam("character_path_");
  base::Status status = base::Param::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("RecognizerPostParam::serialize failed\n");
    return status;
  }

  json.AddMember("character_path_",
                 rapidjson::Value(character_path_.c_str(), allocator),
                 allocator);
  json.AddMember("version_", version_, allocator);
  json.AddMember("rec_thresh_", rec_thresh_, allocator);

  return base::kStatusCodeOk;
}

base::Status RecognizerPostParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("character_path_") && json["character_path_"].IsString()) {
    character_path_ = json["character_path_"].GetString();
  }
  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }

  if (json.HasMember("rec_thresh_") && json["rec_thresh_"].IsFloat()) {
    rec_thresh_ = json["rec_thresh_"].GetFloat();
  }

  return base::kStatusCodeOk;
}

std::vector<std::string> ReadDict(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  while (getline(in, line)) {
    m_vec.push_back(line);
  }
  m_vec.insert(m_vec.begin(), "#");  // blank char for ctc
  m_vec.push_back(" ");
  return m_vec;
}

std::vector<int> ArgSort(const std::vector<float>& array) {
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i) array_index[i] = i;

  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

void RecSingleBatchPostprocessor(const float* out_data,
                                 const std::vector<int64_t>& output_shape,
                                 std::string* text, float* rec_score,
                                 std::vector<std::string> label_list) {
  std::string& str_res = *text;
  float& score = *rec_score;
  score = 0.f;
  int argmax_idx;
  int last_index = 0;
  int count = 0;
  float max_value = 0.0f;

  for (int n = 0; n < output_shape[1]; n++) {
    argmax_idx = int(
        std::distance(&out_data[n * output_shape[2]],
                      std::max_element(&out_data[n * output_shape[2]],
                                       &out_data[(n + 1) * output_shape[2]])));

    max_value = float(*std::max_element(&out_data[n * output_shape[2]],
                                        &out_data[(n + 1) * output_shape[2]]));
    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      if (argmax_idx > label_list.size()) {
        return;
      }
      str_res += label_list[argmax_idx];
    }
    last_index = argmax_idx;
  }
  score /= (count + 1e-6);
  if (count == 0 || std::isnan(score)) {
    score = 0.f;
  }
  return;
}

base::Status RecognizerPreProcess::run() {
  RecognizerPreProcessParam* tmp_param =
      dynamic_cast<RecognizerPreProcessParam*>(param_.get());

  auto edge_ptr = inputs_[0]->get<OCRResult>(this);
  auto detector_result = static_cast<OCRResult*>(edge_ptr);
  int rec_batch_size = tmp_param->rec_batch_size_;
  bool static_shape_infer_ = false;

  std::vector<float> width_list;
  for (const auto& img : detector_result->image_list_) {
    width_list.push_back(float(img.cols) / float(img.rows));
  }

  // 2. 用 ArgSort 得到排序后的索引
  std::vector<int> indices = ArgSort(width_list);

  std::vector<int32_t>* cls_labels_ptr = &detector_result->cls_labels_;
  std::vector<float>* cls_scores_ptr = &detector_result->cls_scores_;

  std::vector<std::string>* text_ptr = &detector_result->text_;
  std::vector<float>* rec_scores_ptr = &detector_result->rec_scores_;
  int batch_size = detector_result->image_list_.size();
  int img_c = tmp_param->rec_image_shape_[0];
  int img_h = tmp_param->rec_image_shape_[1];
  int img_w = tmp_param->rec_image_shape_[2];

  float max_wh_ratio_global = 0.0f;
  for (auto& img : detector_result->image_list_) {
    float wh_ratio = float(img.cols) / float(img.rows);
    if (wh_ratio > max_wh_ratio_global) max_wh_ratio_global = wh_ratio;
  }
  int max_img_w = int(img_h * max_wh_ratio_global);
  // int max_img_w = 584;
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = tmp_param->data_type_;
  desc.data_format_ = tmp_param->data_format_;
  desc.shape_ = {batch_size, img_c, img_h, max_img_w};
  device::Tensor* dst = outputs_[0]->create(device, desc);

  for (size_t start_index = 0;
       start_index < detector_result->image_list_.size();
       start_index += rec_batch_size) {
    size_t end_index = std::min(start_index + rec_batch_size,
                                detector_result->image_list_.size());

    std::vector<cv::Mat> mats(end_index - start_index);

    max_wh_ratio_global = img_w * 1.0 / img_h;
    float ori_wh_ratio;

    for (size_t i = start_index; i < end_index; ++i) {
      size_t real_index = i;
      if (indices.size() != 0) {
        real_index = indices[i];
      }
      mats[i - start_index] = detector_result->image_list_.at(real_index);
    }
    for (size_t i = 0; i < mats.size(); i++) {
      cv::Mat* mat = &(mats[i]);
      ori_wh_ratio = mat->cols * 1.0 / mat->rows;
      max_wh_ratio_global = std::max(max_wh_ratio_global, ori_wh_ratio);
    }
  }

  for (size_t start_index = 0;
       start_index < detector_result->image_list_.size();
       start_index += rec_batch_size) {
    size_t end_index = std::min(start_index + rec_batch_size,
                                detector_result->image_list_.size());

    std::vector<cv::Mat> mats(end_index - start_index);

    float max_wh_ratio = img_w * 1.0 / img_h;
    float ori_wh_ratio;

    for (size_t i = start_index; i < end_index; ++i) {
      size_t real_index = i;
      if (indices.size() != 0) {
        real_index = indices[i];
      }
      mats[i - start_index] = detector_result->image_list_.at(real_index);
    }
    for (size_t i = 0; i < mats.size(); i++) {
      cv::Mat* mat = &(mats[i]);
      ori_wh_ratio = mat->cols * 1.0 / mat->rows;
      max_wh_ratio = std::max(max_wh_ratio, ori_wh_ratio);
    }
    for (size_t i = 0; i < mats.size(); i++) {
      cv::Mat* mat = &(mats[i]);

      cv::Mat tmp_cvt;
      if (tmp_param->src_pixel_type_ != tmp_param->dst_pixel_type_) {
        base::CvtColorType cvt_type = base::calCvtColorType(
            tmp_param->src_pixel_type_, tmp_param->dst_pixel_type_);
        if (cvt_type == base::kCvtColorTypeNotSupport) {
          NNDEPLOY_LOGE("cvtColor type not support");
          return base::kStatusCodeErrorNotSupport;
        }
        int cv_cvt_type =
            preprocess::OpenCvConvert::convertFromCvtColorType(cvt_type);
        cv::cvtColor(*mat, tmp_cvt, cv_cvt_type);
      } else {
        tmp_cvt = *mat;
      }

      int img_h_tmp = tmp_param->rec_image_shape_[1];
      int img_w_tmp = tmp_param->rec_image_shape_[2];
      if (!static_shape_infer_) {
        img_w_tmp = int(img_h_tmp * max_wh_ratio);
        float ratio = float(mat->cols) / float(mat->rows);

        int resize_w;
        if (ceilf(img_h_tmp * ratio) > img_w_tmp) {
          resize_w = img_w_tmp;
        } else {
          resize_w = int(ceilf(img_h_tmp * ratio));
        }

        cv::Mat tmp_resize;
        if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
          int interp_type = preprocess::OpenCvConvert::convertFromInterpType(
              tmp_param->interp_type_);
          cv::resize(tmp_cvt, tmp_resize, cv::Size(resize_w, img_h_tmp), 0.0,
                     0.0, interp_type);
        } else {
          tmp_resize = tmp_cvt;
        }

        cv::Mat tmp_pad;
        if (tmp_resize.cols < img_w_tmp) {
          tmp_param->top_ = 0;
          tmp_param->left_ = 0;
          tmp_param->bottom_ = 0;
          tmp_param->right_ = img_w_tmp - tmp_resize.cols;
          preprocess::OpenCvUtil::copyMakeBorder(
              tmp_resize, tmp_pad, tmp_param->top_, tmp_param->bottom_,
              tmp_param->left_, tmp_param->right_, tmp_param->border_type_,
              tmp_param->border_val_);
        } else {
          tmp_pad = tmp_resize;
        }
        preprocess::OpenCvConvert::convertToBatchTensor(
            tmp_pad, dst, true, tmp_param->scale_, tmp_param->mean_,
            tmp_param->std_, start_index + i);
      }
    }
  }

  int n = dst->getBatch();
  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();
  outputs_[0]->notifyWritten(dst);

  return base::kStatusCodeOk;
}

base::Status RecognizerPostProcess::run() {
  RecognizerPostParam* param = dynamic_cast<RecognizerPostParam*>(param_.get());
  float rec_thresh_ = param->rec_thresh_;
  std::string& character_path = param->character_path_;
  // 访问 PostProcess -> character_dict
  std::vector<std::string> label_list;
  std::ifstream ifs(character_path);
  if (!ifs.is_open()) {
    return base::kStatusCodeErrorIO;
  }
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  ifs.close();
  rapidjson::Document recognizer_config;
  recognizer_config.Parse(content.c_str());
  if (recognizer_config.HasParseError()) {
    return base::kStatusCodeErrorInvalidParam;
  }

  if (recognizer_config.HasMember("PostProcess") &&
      recognizer_config["PostProcess"].IsObject()) {
    const auto& post = recognizer_config["PostProcess"];
    if (post.HasMember("character_dict") && post["character_dict"].IsArray()) {
      const auto& dict = post["character_dict"];
      for (auto& v : dict.GetArray()) {
        if (v.IsString()) {
          label_list.push_back(v.GetString());
        }
      }
    }
  }

  label_list.emplace_back(std::string(" "));
  label_list.insert(label_list.begin(), "blank");
  device::Tensor* tensor = inputs_[0]->getTensor(this);
  float* data = (float*)tensor->getData();

  int batch = tensor->getShapeIndex(0);
  int channel = tensor->getShapeIndex(1);
  int height = tensor->getShapeIndex(2);  // DBNet输出 shape = [batch, 1, H, W]
  int width = tensor->getShapeIndex(3);

  int h = tensor->getShapeIndex(2) > 0 ? tensor->getShapeIndex(2) : 1;
  int w = tensor->getShapeIndex(3) > 0 ? tensor->getShapeIndex(3) : 1;

  const float* out_data = (const float*)tensor->getData();

  std::vector<std::string> texts(batch);
  std::vector<float> rec_scores(batch);
  std::vector<std::string> filtered_texts;
  std::vector<float> filtered_scores;
  const std::vector<int64_t> output_shape = {batch, channel, height, width};
  size_t length1 = channel * height;

  for (int i = 0; i < batch; i++) {
    const float* data_ptr = out_data + i * length1;
    RecSingleBatchPostprocessor(data_ptr, output_shape, &texts[i],
                                &rec_scores[i], label_list);
  }

  OCRResult* ocr_result = new OCRResult();
  filterTexts(texts, rec_scores, rec_thresh_, filtered_texts, filtered_scores);
  ocr_result->text_ = filtered_texts;
  ocr_result->rec_scores_ = filtered_scores;

  // 打印结果
  outputs_[0]->set(ocr_result, false);
  return base::kStatusCodeOk;
}
REGISTER_NODE("nndeploy::ocr::RecognizerPostProcess", RecognizerPostProcess);
REGISTER_NODE("nndeploy::ocr::RecognizerPreProcess", RecognizerPreProcess);
REGISTER_NODE("nndeploy::ocr::RecognizerGraph", RecognizerGraph);

}  // namespace ocr
}  // namespace nndeploy
