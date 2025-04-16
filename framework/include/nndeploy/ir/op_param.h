
#ifndef _NNDEPLOY_IR_OP_PARAM_H_
#define _NNDEPLOY_IR_OP_PARAM_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/rapidjson_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/device/tensor.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace nndeploy {
namespace ir {
/**
 * @brief 算子类型
 * 算子分类
 * # unary
 *  ## activation
 *  ## math
 * # binary - broadcast and elementwise
 * # reduce
 * # computation intensive
 *  ## conv
 *  ## matmul
 * # shape
 *  ## concat
 *  ## split
 * # normalization
 * # pooling
 *
 * deeplink的分类
 *  1. Convolution类
    2. Pooling类
    3. Pad类
    4. Loss类
    5. Norm类
    6. Activation类
    7. Dropout类
    8. Optimizer类
    9. Communication类
    10. Interpolate类
    1. BLAS类
    2. Linalg类
    3. Permute类，对真实内存做数据重排，典型算子为roll、concat等
    # 相比Permute类算子，其没有对真实内存数据做操作和移动，
    # 只是指针信息或者Copy，包括reshape和Indexing等
    4. View/Copy类，
    5. Advanced Indexing类
    6. Distribution类
    7. Sort类
 */
enum OpType : int {
  kOpTypeNet = 0x0000,

  kOpTypeAbs,
  kOpTypeAdd,
  kOpTypeAcos,
  kOpTypeAdam,
  kOpTypeAnd,
  kOpTypeArgMax,
  kOpTypeArgMin,
  kOpTypeAsin,
  kOpTypeAtan,
  kOpTypeAveragePool,
  kOpTypeBatchNormalization,
  kOpTypeCast,
  kOpTypeCeil,
  kOpTypeClip,
  kOpTypeConcat,
  kOpTypeConstant,
  kOpTypeConv,
  kOpTypeCos,
  kOpTypeCosh,
  kOpTypeDepthToSpace,
  kOpTypeDequantizeLinear,
  kOpTypeDet,
  kOpTypeDiv,
  kOpTypeDropout,
  kOpTypeEinsum,
  kOpTypeElu,
  kOpTypeEqual,
  kOpTypeErf,
  kOpTypeExp,
  kOpTypeFlatten,
  kOpTypeFloor,
  kOpTypeGather,
  kOpTypeGemm,
  kOpTypeGlobalAveragePool,
  kOpTypeGlobalLpPool,
  kOpTypeGlobalMaxPool,
  kOpTypeGreater,
  kOpTypeHardSigmoid,
  kOpTypeIdentity,
  kOpTypeIf,
  kOpTypeImageScaler,
  kOpTypeInstanceNormalization,
  kOpTypeIsInf,
  kOpTypeIsNaN,
  kOpTypeLRN,
  kOpTypeLSTM,
  kOpTypeLeakyRelu,
  kOpTypeLess,
  kOpTypeLog,
  kOpTypeLogSoftmax,
  kOpTypeLoop,
  kOpTypeLpNormalization,
  kOpTypeLpPool,
  kOpTypeMatMul,
  kOpTypeMatMulInteger,
  kOpTypeMax,
  kOpTypeMaxPool,
  kOpTypeMaxRoiPool,
  kOpTypeMaxUnpool,
  kOpTypeMean,
  kOpTypeMin,
  kOpTypeMod,
  kOpTypeMomentum,
  kOpTypeMul,
  kOpTypeMultinomial,
  kOpTypeNeg,
  kOpTypeNegLogSoftmax,
  kOpTypeNonMaxSuppression,
  kOpTypeNonZero,
  kOpTypeNot,
  kOpTypeOneHot,
  kOpTypeOnesLike,
  kOpTypeOr,
  kOpTypePad,
  kOpTypePow,
  kOpTypePRelu,
  kOpTypeQLinearConv,
  kOpTypeQLinearMatMul,
  kOpTypeQuantizeLinear,
  kOpTypeRNN,
  kOpTypeRandomNormal,
  kOpTypeRandomNormalLike,
  kOpTypeRandomUniform,
  kOpTypeRandomUniformLike,
  kOpTypeRange,
  kOpTypeReciprocal,
  kOpTypeReduceL1,
  kOpTypeReduceL2,
  kOpTypeReduceLogSum,
  kOpTypeReduceLogSumExp,
  kOpTypeReduceMax,
  kOpTypeReduceMean,
  kOpTypeReduceMin,
  kOpTypeReduceProd,
  kOpTypeReduceSum,
  kOpTypeReduceSumSquare,
  kOpTypeRelu,
  kOpTypeReshape,
  kOpTypeResize,
  kOpTypeReverseSequence,
  kOpTypeRoiAlign,
  kOpTypeRound,
  kOpTypeScale,
  kOpTypeScan,
  kOpTypeScatter,
  kOpTypeSelu,
  kOpTypeSequenceAt,
  kOpTypeSequenceConstruct,
  kOpTypeSequenceEmpty,
  kOpTypeSequenceErase,
  kOpTypeSequenceInsert,
  kOpTypeSequenceLength,
  kOpTypeShape,
  kOpTypeShrink,
  kOpTypeSigmoid,
  kOpTypeSign,
  kOpTypeSin,
  kOpTypeSinh,
  kOpTypeSize,
  kOpTypeSlice,
  kOpTypeSoftmax,
  kOpTypeSoftplus,
  kOpTypeSoftsign,
  kOpTypeSpaceToDepth,
  kOpTypeSplit,
  kOpTypeSqrt,
  kOpTypeSqueeze,
  kOpTypeSub,
  kOpTypeSum,
  kOpTypeTan,
  kOpTypeTanh,
  kOpTypeTfIdf,
  kOpTypeThresholdedRelu,
  kOpTypeTile,
  kOpTypeTopK,
  kOpTypeTranspose,
  kOpTypeUnsqueeze,
  kOpTypeUpsample,
  kOpTypeWhere,
  kOpTypeXor,

  // TODO: @Leonisux:
  // 1. 增加llama的算子类型
  kOpTypeRMSNorm,
  kOpTypeEmbedding,

  kOpTypeNone,
};

NNDEPLOY_CC_API std::string opTypeToString(OpType op_type);

NNDEPLOY_CC_API OpType stringToOpType(const std::string &op_type_name);

/**
 * @brief 算子参数的创建类
 *
 */
class OpParamCreator {
 public:
  virtual ~OpParamCreator() {};
  virtual std::shared_ptr<base::Param> createOpParam(OpType type) = 0;
};

/**
 * @brief 算子参数的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpParamCreator : public OpParamCreator {
  virtual std::shared_ptr<base::Param> createOpParam(OpType type) {
    return std::make_shared<T>();
  }
};

/**
 * @brief Get the Global base::Param Creator Map object
 *
 * @return std::map<OpType, std::shared_ptr<OpParamCreator>>&
 */
std::map<OpType, std::shared_ptr<OpParamCreator>> &getGlobalOpParamCreatorMap();

/**
 * @brief 算子参数的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpParamRegister {
 public:
  explicit TypeOpParamRegister(OpType type) {
    getGlobalOpParamCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a base::Param object
 *
 * @param type
 * @return std::shared_ptr<base::Param>
 */
extern NNDEPLOY_CC_API std::shared_ptr<base::Param> createOpParam(
    OpType op_type);

#define REGISTER_OP_PARAM_IMPLEMENTION(op_type, op_param_class) \
  TypeOpParamRegister<TypeOpParamCreator<op_param_class>>       \
      g_##op_type##_##op_param_class##_register(op_type);

/**
 * @brief
 *
 * @note 按照key,value的形式序列化，每个元素以逗号隔开，具体形式如下
 *
 * reserved_param_:{key,value;key,value;},reserved_:value
 */
class NNDEPLOY_CC_API OpParam : public base::Param {
 public:
  OpParam() : base::Param() {};
  virtual ~OpParam() {};

  PARAM_COPY(OpParam)
  PARAM_COPY_TO(OpParam)

 public:
  // 保留字段,也可以充void *使用
  size_t reserved_;
};

class NNDEPLOY_CC_API BatchNormalizationParam : public OpParam {
 public:
  BatchNormalizationParam() : OpParam() {};
  virtual ~BatchNormalizationParam() {};

  PARAM_COPY(BatchNormalizationParam)
  PARAM_COPY_TO(BatchNormalizationParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("epsilon_", epsilon_, allocator);
    json.AddMember("momentum_", momentum_, allocator);
    json.AddMember("training_mode_", training_mode_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("epsilon_")) {
      epsilon_ = json["epsilon_"].GetFloat();
    } else {
      epsilon_ = 1e-05;  // 默认值
    }

    if (json.HasMember("momentum_")) {
      momentum_ = json["momentum_"].GetFloat();
    } else {
      momentum_ = 0.9;  // 默认值
    }

    if (json.HasMember("training_mode_")) {
      training_mode_ = json["training_mode_"].GetInt();
    } else {
      training_mode_ = 0;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  // The epsilon value to use to avoid division by zero.
  float epsilon_ = 1e-05f;
  // Factor used in computing the running mean and variance.e.g., running_mean =
  // running_mean * momentum + mean * (1 - momentum).
  float momentum_ = 0.9f;
  int training_mode_ = 0;
};

class ConcatParam : public OpParam {
 public:
  ConcatParam() : OpParam() {};
  virtual ~ConcatParam() {};

  PARAM_COPY(ConcatParam)
  PARAM_COPY_TO(ConcatParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("axis_", axis_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("axis_")) {
      axis_ = json["axis_"].GetInt();
    } else {
      axis_ = 1;
    }

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 1;  // 拼接的维度
};

class NNDEPLOY_CC_API ConvParam : public OpParam {
 public:
  // 构造函数
  ConvParam() : OpParam() {}
  virtual ~ConvParam() {}

  PARAM_COPY(ConvParam)
  PARAM_COPY_TO(ConvParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("auto_pad_", rapidjson::Value(auto_pad_.c_str(), allocator),
                   allocator);
    json.AddMember("dilations_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < dilations_.size(); ++i) {
      json["dilations_"].PushBack(dilations_[i], allocator);
    }
    json.AddMember("group_", group_, allocator);
    json.AddMember("kernel_shape_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
      json["kernel_shape_"].PushBack(kernel_shape_[i], allocator);
    }
    json.AddMember("pads_", rapidjson::Value(rapidjson::kArrayType), allocator);
    for (size_t i = 0; i < pads_.size(); ++i) {
      json["pads_"].PushBack(pads_[i], allocator);
    }
    json.AddMember("strides_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < strides_.size(); ++i) {
      json["strides_"].PushBack(strides_[i], allocator);
    }
    json.AddMember(
        "activate_op_",
        rapidjson::Value(opTypeToString(activate_op_).c_str(), allocator),
        allocator);
    if (activate_op_ != kOpTypeNone && fused_op_param_ != nullptr) {
      rapidjson::Value op_desc_json(rapidjson::kObjectType);
      fused_op_param_->serialize(op_desc_json, allocator);
      json.AddMember("fused_op_param_", op_desc_json, allocator);
    }
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("auto_pad_")) {
      auto_pad_ = json["auto_pad_"].GetString();
    } else {
      auto_pad_ = "NOTSET";
    }

    if (json.HasMember("dilations_")) {
      dilations_.clear();
      for (size_t i = 0; i < json["dilations_"].Size(); ++i) {
        dilations_.push_back(json["dilations_"][i].GetInt());
      }
    } else {
      dilations_ = {1, 1};
    }

    if (json.HasMember("group_")) {
      group_ = json["group_"].GetInt();
    } else {
      group_ = 1;
    }

    if (json.HasMember("kernel_shape_")) {
      kernel_shape_.clear();
      for (size_t i = 0; i < json["kernel_shape_"].Size(); ++i) {
        kernel_shape_.push_back(json["kernel_shape_"][i].GetInt());
      }
    } else {
      kernel_shape_.clear();
    }

    if (json.HasMember("pads_")) {
      pads_.clear();
      for (size_t i = 0; i < json["pads_"].Size(); ++i) {
        pads_.push_back(json["pads_"][i].GetInt());
      }
    } else {
      pads_ = {0, 0, 0, 0};
    }

    if (json.HasMember("strides_")) {
      strides_.clear();
      for (size_t i = 0; i < json["strides_"].Size(); ++i) {
        strides_.push_back(json["strides_"][i].GetInt());
      }
    } else {
      strides_ = {1, 1};
    }

    if (json.HasMember("activate_op_")) {
      activate_op_ = stringToOpType(json["activate_op_"].GetString());
      fused_op_param_ = createOpParam(activate_op_);
      if (json.HasMember("fused_op_param_")) {
        fused_op_param_->deserialize(json["fused_op_param_"]);
      }
    } else {
      activate_op_ = kOpTypeNone;
    }

    return base::kStatusCodeOk;
  }

 public:
  // 自动填充方式
  std::string auto_pad_ = "NOTSET";
  // 扩张系数
  std::vector<int> dilations_ = {1, 1};
  // 组数
  int group_ = 1;
  // 卷积核大小
  std::vector<int> kernel_shape_;
  // 填充大小
  std::vector<int> pads_ = {0, 0, 0, 0};
  // 卷积步长
  std::vector<int> strides_ = {1, 1};

  // 服务与算子融合
  OpType activate_op_ = kOpTypeNone;
  // OpParam* fused_op_param_ = nullptr;
  std::shared_ptr<base::Param> fused_op_param_ = nullptr;
};
// MaxPool 参数类
class NNDEPLOY_CC_API MaxPoolParam : public OpParam {
 public:
  MaxPoolParam() : OpParam() {}
  virtual ~MaxPoolParam() {}

  PARAM_COPY(MaxPoolParam)
  PARAM_COPY_TO(MaxPoolParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("auto_pad_", rapidjson::Value(auto_pad_.c_str(), allocator),
                   allocator);
    json.AddMember("ceil_mode_", ceil_mode_, allocator);
    rapidjson::Value dilations_array(rapidjson::kArrayType);
    for (size_t i = 0; i < dilations_.size(); ++i) {
      dilations_array.PushBack(dilations_[i], allocator);
    }
    json.AddMember("dilations_", dilations_array, allocator);

    rapidjson::Value kernel_shape_array(rapidjson::kArrayType);
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
      kernel_shape_array.PushBack(kernel_shape_[i], allocator);
    }
    json.AddMember("kernel_shape_", kernel_shape_array, allocator);

    rapidjson::Value pads_array(rapidjson::kArrayType);
    for (size_t i = 0; i < pads_.size(); ++i) {
      pads_array.PushBack(pads_[i], allocator);
    }
    json.AddMember("pads_", pads_array, allocator);

    json.AddMember("storage_order_", storage_order_, allocator);

    rapidjson::Value strides_array(rapidjson::kArrayType);
    for (size_t i = 0; i < strides_.size(); ++i) {
      strides_array.PushBack(strides_[i], allocator);
    }
    json.AddMember("strides_", strides_array, allocator);

    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("auto_pad_")) {
      auto_pad_ = json["auto_pad_"].GetString();
    } else {
      auto_pad_ = "NOTSET";
    }

    if (json.HasMember("ceil_mode_")) {
      ceil_mode_ = json["ceil_mode_"].GetInt();
    } else {
      ceil_mode_ = 0;
    }

    if (json.HasMember("dilations_")) {
      dilations_.clear();
      for (size_t i = 0; i < json["dilations_"].Size(); ++i) {
        dilations_.push_back(json["dilations_"][i].GetInt());
      }
    } else {
      dilations_ = {1, 1};
    }

    if (json.HasMember("kernel_shape_")) {
      kernel_shape_.clear();
      for (size_t i = 0; i < json["kernel_shape_"].Size(); ++i) {
        kernel_shape_.push_back(json["kernel_shape_"][i].GetInt());
      }
    } else {
      kernel_shape_.clear();
    }

    if (json.HasMember("pads_")) {
      pads_.clear();
      for (size_t i = 0; i < json["pads_"].Size(); ++i) {
        pads_.push_back(json["pads_"][i].GetInt());
      }
    } else {
      pads_ = {0, 0, 0, 0};
    }

    if (json.HasMember("storage_order_")) {
      storage_order_ = json["storage_order_"].GetInt();
    } else {
      storage_order_ = 0;
    }

    if (json.HasMember("strides_")) {
      strides_.clear();
      for (size_t i = 0; i < json["strides_"].Size(); ++i) {
        strides_.push_back(json["strides_"][i].GetInt());
      }
    } else {
      strides_ = {1, 1};
    }

    return base::kStatusCodeOk;
  }

 public:
  std::string auto_pad_ = "NOTSET";       // 自动填充方式
  int ceil_mode_ = 0;                     // 是否向上取整
  std::vector<int> dilations_ = {1, 1};   // 扩张系数
  std::vector<int> kernel_shape_;         // 池化核大小
  std::vector<int> pads_ = {0, 0, 0, 0};  // 填充大小
  int storage_order_ = 0;                 // 存储顺序
  std::vector<int> strides_ = {1, 1};     // 步长
};

// Reshape 参数类
class NNDEPLOY_CC_API ReshapeParam : public OpParam {
 public:
  ReshapeParam() : OpParam() {}
  virtual ~ReshapeParam() {}

  PARAM_COPY(ReshapeParam)
  PARAM_COPY_TO(ReshapeParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("allowzero_", allowzero_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("allowzero_")) {
      allowzero_ = json["allowzero_"].GetInt();
    } else {
      allowzero_ = 0;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  int allowzero_ = 0;  // 是否允许0
};

// Resize 参数类 - opset 18~19
class NNDEPLOY_CC_API ResizeParam : public OpParam {
 public:
  ResizeParam() : OpParam() {}
  virtual ~ResizeParam() {}

  PARAM_COPY(ResizeParam)
  PARAM_COPY_TO(ResizeParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("antialias_", antialias_, allocator);
    json.AddMember("axes_", axes_, allocator);
    json.AddMember(
        "coordinate_transformation_mode_",
        rapidjson::Value(coordinate_transformation_mode_.c_str(), allocator),
        allocator);
    json.AddMember("cubic_coeff_a_", cubic_coeff_a_, allocator);
    json.AddMember("exclude_outside_", exclude_outside_, allocator);
    json.AddMember("extrapolation_value_", extrapolation_value_, allocator);
    json.AddMember(
        "keep_aspect_ratio_policy_",
        rapidjson::Value(keep_aspect_ratio_policy_.c_str(), allocator),
        allocator);
    json.AddMember("mode_", rapidjson::Value(mode_.c_str(), allocator),
                   allocator);
    json.AddMember("nearest_mode_",
                   rapidjson::Value(nearest_mode_.c_str(), allocator),
                   allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("antialias_")) {
      antialias_ = json["antialias_"].GetInt();
    } else {
      antialias_ = 0;  // 默认值
    }

    if (json.HasMember("axes_")) {
      axes_ = json["axes_"].GetInt();
    } else {
      axes_ = INT_MAX;  // 默认值
    }

    if (json.HasMember("coordinate_transformation_mode_")) {
      coordinate_transformation_mode_ =
          json["coordinate_transformation_mode_"].GetString();
    } else {
      coordinate_transformation_mode_ = "half_pixel";  // 默认值
    }

    if (json.HasMember("cubic_coeff_a_")) {
      cubic_coeff_a_ = json["cubic_coeff_a_"].GetFloat();
    } else {
      cubic_coeff_a_ = -0.75;  // 默认值
    }

    if (json.HasMember("exclude_outside_")) {
      exclude_outside_ = json["exclude_outside_"].GetInt();
    } else {
      exclude_outside_ = 0;  // 默认值
    }

    if (json.HasMember("extrapolation_value_")) {
      extrapolation_value_ = json["extrapolation_value_"].GetFloat();
    } else {
      extrapolation_value_ = -0.0;  // 默认值
    }

    if (json.HasMember("keep_aspect_ratio_policy_")) {
      keep_aspect_ratio_policy_ = json["keep_aspect_ratio_policy_"].GetString();
    } else {
      keep_aspect_ratio_policy_ = "stretch";  // 默认值
    }

    if (json.HasMember("mode_")) {
      mode_ = json["mode_"].GetString();
    } else {
      mode_ = "nearest";  // 默认值
    }

    if (json.HasMember("nearest_mode_")) {
      nearest_mode_ = json["nearest_mode_"].GetString();
    } else {
      nearest_mode_ = "round_prefer_floor";  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  int antialias_ = 0;
  int axes_ = INT_MAX;  // 轴，当为INT_MAX时，表示未设置
  std::string coordinate_transformation_mode_ = "half_pixel";
  float cubic_coeff_a_ = -0.75;
  int exclude_outside_ = 0;
  float extrapolation_value_ = -0.0;
  std::string keep_aspect_ratio_policy_ = "stretch";
  std::string mode_ = "nearest";
  std::string nearest_mode_ = "round_prefer_floor";
};

// Softmax 参数类
class NNDEPLOY_CC_API SoftmaxParam : public OpParam {
 public:
  SoftmaxParam() : OpParam() {}
  virtual ~SoftmaxParam() {}

  PARAM_COPY(SoftmaxParam)
  PARAM_COPY_TO(SoftmaxParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("axis_", axis_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("axis_")) {
      axis_ = json["axis_"].GetInt();
    } else {
      axis_ = -1;
    }

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = -1;  // 应用 Softmax 的轴
};

// Split 参数类
class NNDEPLOY_CC_API SplitParam : public OpParam {
 public:
  SplitParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~SplitParam() {}

  PARAM_COPY(SplitParam)
  PARAM_COPY_TO(SplitParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("axis_", axis_, allocator);
    json.AddMember("num_outputs_", num_outputs_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("axis_")) {
      axis_ = json["axis_"].GetInt();
    } else {
      axis_ = 0;  // 默认值
    }

    if (json.HasMember("num_outputs_")) {
      num_outputs_ = json["num_outputs_"].GetInt();
    } else {
      num_outputs_ = INT_MAX;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 0;               // 分割轴
  int num_outputs_ = INT_MAX;  // 分割数
};

// Transpose 参数类
class NNDEPLOY_CC_API TransposeParam : public OpParam {
 public:
  TransposeParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~TransposeParam() {}

  PARAM_COPY(TransposeParam)
  PARAM_COPY_TO(TransposeParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    rapidjson::Value permArray(rapidjson::kArrayType);
    for (size_t i = 0; i < perm_.size(); ++i) {
      permArray.PushBack(perm_[i], allocator);
    }
    json.AddMember("perm_", permArray, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("perm_")) {
      perm_.clear();
      for (size_t i = 0; i < json["perm_"].Size(); ++i) {
        perm_.push_back(json["perm_"][i].GetInt());
      }
    } else {
      perm_.clear();  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  std::vector<int> perm_;
};

// TODO: @Leonisux:
// 补充llama的算子的参数
// RMSNorm 参数类
class NNDEPLOY_CC_API RMSNormParam : public OpParam {
 public:
  RMSNormParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~RMSNormParam() {}

  PARAM_COPY(RMSNormParam)
  PARAM_COPY_TO(RMSNormParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("eps_", eps_, allocator);
    json.AddMember("is_last_", is_last_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("eps_")) {
      eps_ = json["eps_"].GetFloat();
    } else {
      eps_ = 1e-6;  // 默认值
    }

    if (json.HasMember("is_last_")) {
      is_last_ = json["is_last_"].GetBool();
    } else {
      is_last_ = false;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  float eps_ = 1e-6;
  bool is_last_ = false;
};

class NNDEPLOY_CC_API FlattenParam : public OpParam {
 public:
  FlattenParam() : OpParam() {};
  virtual ~FlattenParam() {};

  PARAM_COPY(FlattenParam)
  PARAM_COPY_TO(FlattenParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("axis_", axis_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("axis_")) {
      axis_ = json["axis_"].GetInt();
    } else {
      axis_ = 1;
    }

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 1;
};

class NNDEPLOY_CC_API EmbeddingParam : public OpParam {
 public:
  EmbeddingParam() : OpParam() {};
  virtual ~EmbeddingParam() {};

  PARAM_COPY(EmbeddingParam)
  PARAM_COPY_TO(EmbeddingParam)

  // base::Status serialize(rapidjson::Value &json,
  //                        rapidjson::Document::AllocatorType &allocator) {}
  // base::Status deserialize(rapidjson::Value &json) {
  //   return base::kStatusCodeOk;
  // }
};

class NNDEPLOY_CC_API GemmParam : public OpParam {
 public:
  GemmParam() : OpParam() {};
  virtual ~GemmParam() {};

  PARAM_COPY(GemmParam)
  PARAM_COPY_TO(GemmParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("alpha_", alpha_, allocator);
    json.AddMember("beta_", beta_, allocator);
    json.AddMember("trans_a_", trans_a_, allocator);
    json.AddMember("trans_b_", trans_b_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("alpha_")) {
      alpha_ = json["alpha_"].GetFloat();
    } else {
      alpha_ = 1.0;  // 默认值
    }

    if (json.HasMember("beta_")) {
      beta_ = json["beta_"].GetFloat();
    } else {
      beta_ = 1.0;  // 默认值
    }

    if (json.HasMember("trans_a_")) {
      trans_a_ = json["trans_a_"].GetInt();
    } else {
      trans_a_ = 0;  // 默认值
    }

    if (json.HasMember("trans_b_")) {
      trans_b_ = json["trans_b_"].GetInt();
    } else {
      trans_b_ = 0;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  float alpha_ = 1.0;  // 默认值为1.0
  float beta_ = 1.0;   // 默认值为1.0
  int trans_a_ = 0;    // 默认值为0
  int trans_b_ = 0;    // 默认值为0
};

class NNDEPLOY_CC_API QuantizeLinearParam : public OpParam {
 public:
  QuantizeLinearParam() : OpParam() {}
  virtual ~QuantizeLinearParam() {}

  PARAM_COPY(QuantizeLinearParam)
  PARAM_COPY_TO(QuantizeLinearParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("axis_", axis_, allocator);
    json.AddMember("saturate_", saturate_, allocator);
    return base::kStatusCodeOk;
  }

  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("axis_")) {
      axis_ = json["axis_"].GetInt();
    } else {
      axis_ = 1;  // 默认值
    }

    if (json.HasMember("saturate_")) {
      saturate_ = json["saturate_"].GetInt();
    } else {
      saturate_ = 1;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 1;      // 量化维度，默认为1
  int saturate_ = 1;  // 是否饱和，默认为1
};

class NNDEPLOY_CC_API DequantizeLinearParam : public OpParam {
 public:
  DequantizeLinearParam() : OpParam() {}
  virtual ~DequantizeLinearParam() {}

  PARAM_COPY(DequantizeLinearParam)
  PARAM_COPY_TO(DequantizeLinearParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("axis_", axis_, allocator);
    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("axis_")) {
      axis_ = json["axis_"].GetInt();
    } else {
      axis_ = 1;  // 默认值
    }

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 1;  // 反量化维度，默认为1
};

class NNDEPLOY_CC_API QLinearConvParam : public OpParam {
 public:
  // 构造函数
  QLinearConvParam() : OpParam() {}
  virtual ~QLinearConvParam() {}

  PARAM_COPY(QLinearConvParam)
  PARAM_COPY_TO(QLinearConvParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("auto_pad_", rapidjson::Value(auto_pad_.c_str(), allocator),
                   allocator);
    json.AddMember("dilations_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < dilations_.size(); ++i) {
      json["dilations_"].PushBack(dilations_[i], allocator);
    }
    json.AddMember("group_", group_, allocator);
    json.AddMember("kernel_shape_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
      json["kernel_shape_"].PushBack(kernel_shape_[i], allocator);
    }
    json.AddMember("pads_", rapidjson::Value(rapidjson::kArrayType), allocator);
    for (size_t i = 0; i < pads_.size(); ++i) {
      json["pads_"].PushBack(pads_[i], allocator);
    }
    json.AddMember("strides_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < strides_.size(); ++i) {
      json["strides_"].PushBack(strides_[i], allocator);
    }

    return base::kStatusCodeOk;
  }
  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("auto_pad_")) {
      auto_pad_ = json["auto_pad_"].GetString();
    } else {
      auto_pad_ = "NOTSET";
    }

    if (json.HasMember("dilations_")) {
      dilations_.clear();
      for (size_t i = 0; i < json["dilations_"].Size(); ++i) {
        dilations_.push_back(json["dilations_"][i].GetInt());
      }
    } else {
      dilations_ = {1, 1};
    }

    if (json.HasMember("group_")) {
      group_ = json["group_"].GetInt();
    } else {
      group_ = 1;
    }

    if (json.HasMember("kernel_shape_")) {
      kernel_shape_.clear();
      for (size_t i = 0; i < json["kernel_shape_"].Size(); ++i) {
        kernel_shape_.push_back(json["kernel_shape_"][i].GetInt());
      }
    } else {
      kernel_shape_.clear();
    }

    if (json.HasMember("pads_")) {
      pads_.clear();
      for (size_t i = 0; i < json["pads_"].Size(); ++i) {
        pads_.push_back(json["pads_"][i].GetInt());
      }
    } else {
      pads_ = {0, 0, 0, 0};
    }

    if (json.HasMember("strides_")) {
      strides_.clear();
      for (size_t i = 0; i < json["strides_"].Size(); ++i) {
        strides_.push_back(json["strides_"][i].GetInt());
      }
    } else {
      strides_ = {1, 1};
    }

    return base::kStatusCodeOk;
  }

 public:
  // 自动填充方式
  std::string auto_pad_ = "NOTSET";
  // 扩张系数
  std::vector<int> dilations_ = {1, 1};
  // 组数
  int group_ = 1;
  // 卷积核大小
  std::vector<int> kernel_shape_;
  // 填充大小
  std::vector<int> pads_ = {0, 0, 0, 0};
  // 卷积步长
  std::vector<int> strides_ = {1, 1};
};

class NNDEPLOY_CC_API AvaragePoolParam : public OpParam {
 public:
  // 构造函数
  AvaragePoolParam() : OpParam() {}
  virtual ~AvaragePoolParam() {}

  PARAM_COPY(AvaragePoolParam)
  PARAM_COPY_TO(AvaragePoolParam)

  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("auto_pad_", rapidjson::Value(auto_pad_.c_str(), allocator),
                   allocator);
    json.AddMember("ceil_mode_", ceil_mode_, allocator);
    json.AddMember("count_include_pad_",
                   rapidjson::Value(count_include_pad_.c_str(), allocator),
                   allocator);
    json.AddMember("dilations_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < dilations_.size(); ++i) {
      json["dilations_"].PushBack(dilations_[i], allocator);
    }
    json.AddMember("kernel_shape_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
      json["kernel_shape_"].PushBack(kernel_shape_[i], allocator);
    }
    json.AddMember("pads_", rapidjson::Value(rapidjson::kArrayType), allocator);
    for (size_t i = 0; i < pads_.size(); ++i) {
      json["pads_"].PushBack(pads_[i], allocator);
    }
    json.AddMember("strides_", rapidjson::Value(rapidjson::kArrayType),
                   allocator);
    for (size_t i = 0; i < strides_.size(); ++i) {
      json["strides_"].PushBack(strides_[i], allocator);
    }
    return base::kStatusCodeOk;
  }

  base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("auto_pad_")) {
      auto_pad_ = json["auto_pad_"].GetString();
    } else {
      auto_pad_ = "NOTSET";
    }

    if (json.HasMember("ceil_mode_")) {
      ceil_mode_ = json["ceil_mode_"].GetInt();
    } else {
      ceil_mode_ = 0;
    }

    if (json.HasMember("count_include_pad_")) {
      count_include_pad_ = json["count_include_pad_"].GetString();
    } else {
      count_include_pad_ = "EXCLUDE";
    }

    if (json.HasMember("dilations_")) {
      dilations_.clear();
      for (size_t i = 0; i < json["dilations_"].Size(); ++i) {
        dilations_.push_back(json["dilations_"][i].GetInt());
      }
    } else {
      dilations_ = {1, 1};
    }

    if (json.HasMember("kernel_shape_")) {
      kernel_shape_.clear();
      for (size_t i = 0; i < json["kernel_shape_"].Size(); ++i) {
        kernel_shape_.push_back(json["kernel_shape_"][i].GetInt());
      }
    } else {
      kernel_shape_.clear();
    }

    if (json.HasMember("pads_")) {
      pads_.clear();
      for (size_t i = 0; i < json["pads_"].Size(); ++i) {
        pads_.push_back(json["pads_"][i].GetInt());
      }
    } else {
      pads_ = {0, 0, 0, 0};
    }

    if (json.HasMember("strides_")) {
      strides_.clear();
      for (size_t i = 0; i < json["strides_"].Size(); ++i) {
        strides_.push_back(json["strides_"][i].GetInt());
      }
    } else {
      strides_ = {1, 1};
    }

    return base::kStatusCodeOk;
  }

 public:
  // 自动填充方式
  std::string auto_pad_ = "NOTSET";
  // 是否向上取整
  int ceil_mode_ = 0;
  // 计算方式
  std::string count_include_pad_ = "EXCLUDE";
  // 扩张系数
  std::vector<int> dilations_ = {1, 1};
  // 平均池化的核大小
  std::vector<int> kernel_shape_;
  // 填充大小
  std::vector<int> pads_ = {0, 0, 0, 0};
  // 平均池化的步长
  std::vector<int> strides_ = {1, 1};
};

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_OP_PARAM_H_ */
