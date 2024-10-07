
#ifndef _NNDEPLOY_IR_OP_PARAM_H_
#define _NNDEPLOY_IR_OP_PARAM_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/tensor.h"

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

  kOpTypeNone,
};

std::string opTypeToString(OpType op_type);

OpType stringToOpType(const std::string &op_type_name);

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
class OpParam : public base::Param {
 public:
  OpParam() : base::Param() {};
  virtual ~OpParam() {};

  PARAM_COPY(OpParam)
  PARAM_COPY_TO(OpParam)

 public:
  // 保留字段,也可以充void *使用
  size_t reserved_;
};

class BatchNormalizationParam : public OpParam {
 public:
  BatchNormalizationParam() : OpParam() {};
  virtual ~BatchNormalizationParam() {};

  PARAM_COPY(BatchNormalizationParam)
  PARAM_COPY_TO(BatchNormalizationParam)

  base::Status serialize(std::ostream &stream) {
    stream << "epsilon_:" << epsilon_ << ",";
    stream << "momentum_:" << momentum_ << ",";
    stream << "training_mode_:" << training_mode_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 epsilon_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "epsilon_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    epsilon_ = std::stof(token);

    // 读取 momentum_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "momentum_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    momentum_ = std::stof(token);

    // 读取 training_mode_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "training_mode_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    training_mode_ = std::stoi(token);

    return base::kStatusCodeOk;
  }

 public:
  // The epsilon value to use to avoid division by zero.
  float epsilon_ = 1e-05;
  // Factor used in computing the running mean and variance.e.g., running_mean =
  // running_mean * momentum + mean * (1 - momentum).
  float momentum_ = 0.9;
  int training_mode_ = 0;
};

class ConcatParam : public OpParam {
 public:
  ConcatParam() : OpParam() {};
  virtual ~ConcatParam() {};

  PARAM_COPY(ConcatParam)
  PARAM_COPY_TO(ConcatParam)

  base::Status serialize(std::ostream &stream) {
    stream << "axis_:" << axis_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 axis_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "axis_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    axis_ = std::stoi(token);

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 1;  // 拼接的维度
};

class ConvParam : public OpParam {
 public:
  // 构造函数
  ConvParam() : OpParam() {}
  virtual ~ConvParam() {}

  PARAM_COPY(ConvParam)
  PARAM_COPY_TO(ConvParam)

  base::Status serialize(std::ostream &stream) {
    stream << "auto_pad_:" << auto_pad_ << ",";
    stream << "dilations_:[";
    for (size_t i = 0; i < dilations_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << dilations_[i];
    }
    stream << "],";
    stream << "group_:" << group_ << ",";
    stream << "kernel_shape_:[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << kernel_shape_[i];
    }
    stream << "],";
    stream << "pads_:[";
    for (size_t i = 0; i < pads_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << pads_[i];
    }
    stream << "],";
    stream << "strides_:[";
    for (size_t i = 0; i < strides_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << strides_[i];
    }
    stream << "],";
    stream << "is_fusion_op_:" << is_fusion_op_ << ",";
    stream << "activate_op_:" << opTypeToString(activate_op_) << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 auto_pad_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "auto_pad_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, auto_pad_, ','))
      return base::kStatusCodeErrorInvalidValue;

    // 读取 dilations_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "dilations_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    dilations_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        dilations_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        dilations_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 group_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "group_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    group_ = std::stoi(token);

    // 读取 kernel_shape_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "kernel_shape_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    kernel_shape_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        kernel_shape_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        kernel_shape_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 pads_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "pads_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    pads_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        pads_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        pads_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 strides_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "strides_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    strides_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        strides_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        strides_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 is_fusion_op_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "is_fusion_op_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    is_fusion_op_ = (token == "1");

    // 读取 activate_op_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "activate_op_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    activate_op_ = stringToOpType(token);

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

  // 基于onnx扩展的参数
  bool is_fusion_op_ = false;
  OpType activate_op_ = kOpTypeRelu;
};
// MaxPool 参数类
class MaxPoolParam : public OpParam {
 public:
  MaxPoolParam() : OpParam() {}
  virtual ~MaxPoolParam() {}

  PARAM_COPY(MaxPoolParam)
  PARAM_COPY_TO(MaxPoolParam)

  base::Status serialize(std::ostream &stream) {
    stream << "auto_pad_:" << auto_pad_ << ",";
    stream << "ceil_mode_:" << ceil_mode_ << ",";
    stream << "dilations_:[";
    for (size_t i = 0; i < dilations_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << dilations_[i];
    }
    stream << "],";

    stream << "kernel_shape_:[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << kernel_shape_[i];
    }
    stream << "],";

    stream << "pads_:[";
    for (size_t i = 0; i < pads_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << pads_[i];
    }
    stream << "],";

    stream << "storage_order_:" << storage_order_ << ",";
    stream << "strides_:[";
    for (size_t i = 0; i < strides_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << strides_[i];
    }
    stream << "],";

    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 auto_pad_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "auto_pad_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, auto_pad_, ','))
      return base::kStatusCodeErrorInvalidValue;

    // 读取 ceil_mode_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "ceil_mode_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    ceil_mode_ = std::stoi(token);

    // 读取 dilations_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "dilations_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    dilations_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        dilations_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        dilations_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 kernel_shape_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "kernel_shape_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    kernel_shape_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        kernel_shape_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        kernel_shape_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 pads_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "pads_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    pads_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        pads_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        pads_.push_back(std::stoi(token));
        break;
      }
    }

    // 读取 storage_order_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "storage_order_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    storage_order_ = std::stoi(token);

    // 读取 strides_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "strides_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    strides_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        strides_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        strides_.push_back(std::stoi(token));
        break;
      }
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
class ReshapeParam : public OpParam {
 public:
  ReshapeParam() : OpParam() {}
  virtual ~ReshapeParam() {}

  PARAM_COPY(ReshapeParam)
  PARAM_COPY_TO(ReshapeParam)

  base::Status serialize(std::ostream &stream) {
    stream << "allowzero_:" << allowzero_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 allowzero_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "allowzero_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    allowzero_ = std::stoi(token);

    return base::kStatusCodeOk;
  }

 public:
  int allowzero_ = 0;  // 是否允许0
};

// Resize 参数类 - opset 18~19
class ResizeParam : public OpParam {
 public:
  ResizeParam() : OpParam() {}
  virtual ~ResizeParam() {}

  PARAM_COPY(ResizeParam)
  PARAM_COPY_TO(ResizeParam)

  base::Status serialize(std::ostream &stream) {
    stream << "antialias_:" << antialias_ << ",";
    stream << "axes_:" << axes_ << ",";
    stream << "coordinate_transformation_mode_:"
           << coordinate_transformation_mode_ << ",";
    stream << "cubic_coeff_a_:" << cubic_coeff_a_ << ",";
    stream << "exclude_outside_:" << exclude_outside_ << ",";
    stream << "extrapolation_value_:" << extrapolation_value_ << ",";
    stream << "keep_aspect_ratio_policy_:" << keep_aspect_ratio_policy_ << ",";
    stream << "mode_:" << mode_ << ",";
    stream << "nearest_mode_:" << nearest_mode_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 antialias_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "antialias_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    antialias_ = std::stoi(token);

    // 读取 axes_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "axes_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    axes_ = std::stoi(token);

    // 读取 coordinate_transformation_mode_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "coordinate_transformation_mode_")
      return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    coordinate_transformation_mode_ = token;

    // 读取 cubic_coeff_a_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "cubic_coeff_a_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    cubic_coeff_a_ = std::stof(token);

    // 读取 exclude_outside_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "exclude_outside_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    exclude_outside_ = std::stoi(token);

    // 读取 extrapolation_value_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "extrapolation_value_")
      return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    extrapolation_value_ = std::stof(token);

    // 读取 keep_aspect_ratio_policy_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "keep_aspect_ratio_policy_")
      return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    keep_aspect_ratio_policy_ = token;

    // 读取 mode_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "mode_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    mode_ = token;

    // 读取 nearest_mode_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "nearest_mode_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    nearest_mode_ = token;

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
class SoftmaxParam : public OpParam {
 public:
  SoftmaxParam() : OpParam() {}
  virtual ~SoftmaxParam() {}

  PARAM_COPY(SoftmaxParam)
  PARAM_COPY_TO(SoftmaxParam)

  base::Status serialize(std::ostream &stream) {
    stream << "axis_:" << axis_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 axis_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "axis_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    axis_ = std::stoi(token);

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = -1;  // 应用 Softmax 的轴
};

// Split 参数类
class SplitParam : public OpParam {
 public:
  SplitParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~SplitParam() {}

  PARAM_COPY(SplitParam)
  PARAM_COPY_TO(SplitParam)

  base::Status serialize(std::ostream &stream) {
    stream << "axis_:" << axis_ << ",";
    stream << "num_outputs_:" << num_outputs_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 axis_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "axis_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    axis_ = std::stoi(token);

    // 读取 num_outputs_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "num_outputs_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    num_outputs_ = std::stoi(token);

    return base::kStatusCodeOk;
  }

 public:
  int axis_ = 0;               // 分割轴
  int num_outputs_ = INT_MAX;  // 分割数
};

// Transpose 参数类
class TransposeParam : public OpParam {
 public:
  TransposeParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~TransposeParam() {}

  PARAM_COPY(TransposeParam)
  PARAM_COPY_TO(TransposeParam)

  base::Status serialize(std::ostream &stream) {
    stream << "perm_:[";
    for (size_t i = 0; i < perm_.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << perm_[i];
    }
    stream << "],";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 perm_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "perm_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, '['))
      return base::kStatusCodeErrorInvalidValue;
    perm_.clear();
    while (std::getline(iss, token, ',')) {
      if (token.find(']') == std::string::npos) {
        perm_.push_back(std::stoi(token));
      } else {
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        perm_.push_back(std::stoi(token));
        break;
      }
    }

    return base::kStatusCodeOk;
  }

 public:
  std::vector<int> perm_;
};

// TODO: @Leonisux:
// 补充llama的算子的参数
// RMSNorm 参数类
class RMSNormParam : public OpParam {
 public:
  RMSNormParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~RMSNormParam() {}

  PARAM_COPY(RMSNormParam)
  PARAM_COPY_TO(RMSNormParam)

  base::Status serialize(std::ostream &stream) {
    stream << "eps_:" << eps_ << ",";
    stream << "is_last_:" << is_last_ << ",";
    return base::kStatusCodeOk;
  }
  base::Status deserialize(const std::string &line) {
    std::istringstream iss(line);
    std::string token;

    // 读取 eps_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "eps_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    eps_ = std::stof(token);

    // 读取 is_last_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "is_last_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    is_last_ = (token == "1");

    return base::kStatusCodeOk;
  }

 public:
  float eps_ = 1e-6;
  bool is_last_ = false;
};

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_OP_PARAM_H_ */
