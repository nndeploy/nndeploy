
#ifndef _NNDEPLOY_PREPROCESS_PARAMS_H_
#define _NNDEPLOY_PREPROCESS_PARAMS_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace preprocess {

/**
 * @brief 前处理通常由如下算子组合排列
 * cvtcolor
 * resize
 * padding
 * warp_affine
 * crop
 * nomalize
 * transpose
 * dynamic_shape
 */
class NNDEPLOY_CC_API CvtcolorParam : public base::Param {
 public:
  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
};

class NNDEPLOY_CC_API ResizeParam : public base::Param {
 public:
  base::InterpType interp_type_ = base::kInterpTypeLinear;
  float scale_w_ = 0.0f;
  float scale_h_ = 0.0f;
  int dst_h_ = -1;
  int dst_w_ = -1;
};

class NNDEPLOY_CC_API PaddingParam : public base::Param {
 public:
  base::BorderType border_type_ = base::kBorderTypeConstant;
  int top_ = 0;
  int bottom_ = 0;
  int left_ = 0;
  int right_ = 0;
  base::Scalar2d border_val_ = 0.0;
};

class NNDEPLOY_CC_API WarpAffineParam : public base::Param {
 public:
  float transform_[2][3];
  int dst_w_;
  int dst_h_;

  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  int h_ = -1;
  int w_ = -1;
  bool normalize_ = true;
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  int const_value_ = 114;

  base::InterpType interp_type_ = base::kInterpTypeLinear;
  base::BorderType border_type_ = base::kBorderTypeConstant;
  base::Scalar2d border_val_ = 0.0f;
};

class NNDEPLOY_CC_API CropParam : public base::Param {
 public:
  int top_left_x_ = 0;
  int top_left_y_ = 0;
  int width_ = 0;
  int height_ = 0;
};

class NNDEPLOY_CC_API NomalizeParam : public base::Param {
 public:
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

class NNDEPLOY_CC_API TransposeParam : public base::Param {
 public:
  base::DataFormat src_data_format_ = base::kDataFormatNHWC;
  base::DataFormat dst_data_format_ = base::kDataFormatNCHW;
};

class NNDEPLOY_CC_API DynamicShapeParam : public base::Param {
 public:
  bool is_power_of_n_ = false;
  int n_ = 2;
  int w_align_ = 1;
  int h_align_ = 1;
  base::IntVector min_shape_;
  base::IntVector opt_shape_;
  base::IntVector max_shape_;
};

/**
 * @brief 组合的预处理
 *
 */

class NNDEPLOY_CC_API CvtcolorBnParam : public base::Param {
 public:
  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
  // 数据类型，默认为浮点型
  base::DataType data_type_ = base::dataTypeOf<float>();
  // 数据格式，默认为NCHW（通道数，图像高度，图像宽度）
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  // 是否进行归一化处理
  bool normalize_ = true;
  // 归一化的比例因子，用于将像素值缩放到0-1范围
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  // 归一化处理中的均值，用于数据中心化
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // 归一化处理中的标准差，用于数据标准化
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

class NNDEPLOY_CC_API CvtclorResizeParam : public base::Param {
 public:
  // 源图像的像素类型
  base::PixelType src_pixel_type_;
  // 目标图像的像素类型
  base::PixelType dst_pixel_type_;
  // 图像缩放时使用的插值类型
  base::InterpType interp_type_;
  // 目标输出的高度
  int h_ = -1;
  // 目标输出的宽度
  int w_ = -1;
  // 数据类型，默认为浮点型
  base::DataType data_type_ = base::dataTypeOf<float>();
  // 数据格式，默认为NCHW（通道数，图像高度，图像宽度）
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  // 是否进行归一化处理
  bool normalize_ = true;
  // 归一化的比例因子，用于将像素值缩放到0-1范围
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  // 归一化处理中的均值，用于数据中心化
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // 归一化处理中的标准差，用于数据标准化
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  virtual base::Status serialize(
      rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
    base::Status status = base::Param::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }

    std::string src_pixel_type_str = base::pixelTypeToString(src_pixel_type_);
    json.AddMember("src_pixel_type_",
                   rapidjson::Value(src_pixel_type_str.c_str(), allocator),
                   allocator);
    std::string dst_pixel_type_str = base::pixelTypeToString(dst_pixel_type_);
    json.AddMember("dst_pixel_type_",
                   rapidjson::Value(dst_pixel_type_str.c_str(), allocator),
                   allocator);
    std::string interp_type_str = base::interpTypeToString(interp_type_);
    json.AddMember("interp_type_",
                   rapidjson::Value(interp_type_str.c_str(), allocator),
                   allocator);
    json.AddMember("h_", h_, allocator);
    json.AddMember("w_", w_, allocator);
    std::string data_type_str = base::dataTypeToString(data_type_);
    json.AddMember("data_type_",
                   rapidjson::Value(data_type_str.c_str(), allocator),
                   allocator);
    std::string data_format_str = base::dataFormatToString(data_format_);
    json.AddMember("data_format_",
                   rapidjson::Value(data_format_str.c_str(), allocator),
                   allocator);
    json.AddMember("normalize_", normalize_, allocator);

    rapidjson::Value scale_array(rapidjson::kArrayType);
    rapidjson::Value mean_array(rapidjson::kArrayType);
    rapidjson::Value std_array(rapidjson::kArrayType);
    for (int i = 0; i < 4; i++) {
      scale_array.PushBack(scale_[i], allocator);
      mean_array.PushBack(mean_[i], allocator);
      std_array.PushBack(std_[i], allocator);
    }
    json.AddMember("scale_", scale_array, allocator);
    json.AddMember("mean_", mean_array, allocator);
    json.AddMember("std_", std_array, allocator);

    return base::kStatusCodeOk;
  }

  virtual base::Status deserialize(rapidjson::Value& json) {
    base::Status status = base::Param::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }

    if (json.HasMember("src_pixel_type_") &&
        json["src_pixel_type_"].IsString()) {
      src_pixel_type_ =
          base::stringToPixelType(json["src_pixel_type_"].GetString());
    }
    if (json.HasMember("dst_pixel_type_") &&
        json["dst_pixel_type_"].IsString()) {
      dst_pixel_type_ =
          base::stringToPixelType(json["dst_pixel_type_"].GetString());
    }
    if (json.HasMember("interp_type_") && json["interp_type_"].IsString()) {
      interp_type_ = base::stringToInterpType(json["interp_type_"].GetString());
    }
    if (json.HasMember("h_") && json["h_"].IsInt()) {
      h_ = json["h_"].GetInt();
    }
    if (json.HasMember("w_") && json["w_"].IsInt()) {
      w_ = json["w_"].GetInt();
    }
    if (json.HasMember("data_type_") && json["data_type_"].IsString()) {
      data_type_ = base::stringToDataType(json["data_type_"].GetString());
    }
    if (json.HasMember("data_format_") && json["data_format_"].IsString()) {
      data_format_ = base::stringToDataFormat(json["data_format_"].GetString());
    }
    if (json.HasMember("normalize_") && json["normalize_"].IsBool()) {
      normalize_ = json["normalize_"].GetBool();
    }

    if (json.HasMember("scale_") && json["scale_"].IsArray()) {
      const rapidjson::Value& scale_array = json["scale_"];
      for (int i = 0; i < 4 && i < scale_array.Size(); i++) {
        if (scale_array[i].IsFloat()) {
          scale_[i] = scale_array[i].GetFloat();
        }
      }
    }
    if (json.HasMember("mean_") && json["mean_"].IsArray()) {
      const rapidjson::Value& mean_array = json["mean_"];
      for (int i = 0; i < 4 && i < mean_array.Size(); i++) {
        if (mean_array[i].IsFloat()) {
          mean_[i] = mean_array[i].GetFloat();
        }
      }
    }
    if (json.HasMember("std_") && json["std_"].IsArray()) {
      const rapidjson::Value& std_array = json["std_"];
      for (int i = 0; i < 4 && i < std_array.Size(); i++) {
        if (std_array[i].IsFloat()) {
          std_[i] = std_array[i].GetFloat();
        }
      }
    }

    return base::kStatusCodeOk;
  }
};

/**
 * @brief 组合的预处理
 *
 */
class NNDEPLOY_CC_API CvtclorResizePadParam : public base::Param {
 public:
  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
  base::InterpType interp_type_;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  int h_ = -1;
  int w_ = -1;
  bool normalize_ = true;
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  base::BorderType border_type_ = base::kBorderTypeConstant;
  int top_ = 0;
  int bottom_ = 0;
  int left_ = 0;
  int right_ = 0;
  base::Scalar2d border_val_ = 0.0;
};

/**
 * @brief 组合的预处理
 *
 */
class NNDEPLOY_CC_API CvtColorResizeCropParam : public base::Param {
 public:
  base::PixelType src_pixel_type_;
  base::PixelType dst_pixel_type_;
  base::InterpType interp_type_;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;

  int resize_h_ = -1;
  int resize_w_ = -1;

  bool normalize_ = true;
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  int top_left_x_ = 0;
  int top_left_y_ = 0;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace preprocess
}  // namespace nndeploy

#endif /* _NNDEPLOY_PREPROCESS_PARAMS_H_ */
