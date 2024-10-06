
#ifndef _NNDEPLOY_IR_IR_H_
#define _NNDEPLOY_IR_IR_H_

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
 * @brief 参照并扩充了onnx的格式，描述算子的基本信息
 * # 1. 算子名称
 * # 2. 算子类型
 * # 3. 算子输入
 * # 4. 算子输出
 * # 5. 算子的参数
 */
class NNDEPLOY_CC_API OpDesc {
 public:
  OpDesc();
  OpDesc(OpType op_type);
  OpDesc(const std::string &name, OpType op_type);
  OpDesc(const std::string &name, OpType op_type,
         std::shared_ptr<base::Param> op_param);
  OpDesc(const std::string &name, OpType op_type,
         std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs);
  OpDesc(const std::string &name, OpType op_type,
         std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs,
         std::shared_ptr<base::Param> op_param);
  OpDesc(const std::string &name, OpType op_type,
         std::vector<std::string> &inputs, std::vector<std::string> &outputs);
  OpDesc(const std::string &name, OpType op_type,
         std::vector<std::string> &inputs, std::vector<std::string> &outputs,
         std::shared_ptr<base::Param> op_param);

  virtual ~OpDesc();

  // 序列化
  base::Status serialize(std::ostream &stream) const;
  // 反序列化
  base::Status deserialize(const std::string &line);

 public:
  // 算子名称
  std::string name_;
  // 节点类型
  OpType op_type_;
  // 节点输入 : 包含 input、weight等所有参与计算的数据
  std::vector<std::string> inputs_;
  // 节点输出
  std::vector<std::string> outputs_;
  // 算子参数
  std::shared_ptr<base::Param> op_param_;
};

/**
 * @brief 参照onnx的格式，描述模型或者算子输入输出
 *
 */
class ValueDesc {
 public:
  ValueDesc();
  ValueDesc(const std::string &name);
  ValueDesc(const std::string &name, base::DataType data_type);
  ValueDesc(const std::string &name, base::DataType data_type,
            base::IntVector shape);

  virtual ~ValueDesc();

  // 序列化
  base::Status serialize(std::ostream &stream) const;
  // 反序列化
  base::Status deserialize(const std::string &line);

 public:
  // 名称
  std::string name_;
  // 数据类型
  base::DataType data_type_;
  // 张量形状
  base::IntVector shape_;
};

/**
 * @brief 参照onnx的格式，描述模型的结构
 *
 */
class ModelDesc {
 public:
  ModelDesc();
  virtual ~ModelDesc();

  base::Status dump(std::ostream &oss);

  // 序列化模型结构为文本
  base::Status serializeStructureToText(std::ostream &stream) const;
  // 反序列化文本为模型结构
  base::Status deserializeStructureFromText(
      std::istream &stream, const std::vector<ValueDesc> &input);
  // 序列化模型权重为二进制文件
  base::Status serializeWeightsToBinary(std::ostream &stream) const;
  // 从二进制文件反序列化模型权重
  base::Status deserializeWeightsFromBinary(std::istream &stream);

 public:
  // 描述模型的名称
  std::string name_;
  // 模型输入
  std::vector<std::shared_ptr<ValueDesc>> inputs_;
  // 模型输出
  std::vector<std::shared_ptr<ValueDesc>> outputs_;
  // 模型算子列表
  std::vector<std::shared_ptr<OpDesc>> op_descs_;
  // 模型权重
  std::map<std::string, device::Tensor *> weights_;
  // 模型中间值，一般通常为空，多用于调试
  std::vector<std::shared_ptr<ValueDesc>> values_;
  // 模型块
  std::vector<std::shared_ptr<ModelDesc>> blocks_;
};

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

  // // 序列化
  // base::Status serialize(std::ostream &stream) {
  //   stream << "reserved_param_:";
  //   stream << "{";
  //   for (const auto &param : reserved_param_) {
  //     stream << param.first << "," << "param.second.toString()" << ";";
  //   }
  //   stream << "},";
  //   stream << "reserved_:" << reserved_ << ",";
  //   return base::kStatusCodeOk;
  // }
  // // 反序列化
  // base::Status deserialize(const std::string &line) {
  //   std::istringstream iss(line);
  //   std::string token;

  //   // 读取 reserved_param_
  //   if (!std::getline(iss, token, ':'))
  //     return base::kStatusCodeErrorInvalidValue;
  //   if (token != "reserved_param_") return
  //   base::kStatusCodeErrorInvalidValue; if (!std::getline(iss, token, '{'))
  //     return base::kStatusCodeErrorInvalidValue;
  //   while (std::getline(iss, token, ';')) {
  //     if (token == "}") break;
  //     std::istringstream param_iss(token);
  //     std::string key, value;
  //     if (!std::getline(param_iss, key, ','))
  //       return base::kStatusCodeErrorInvalidValue;
  //     if (!std::getline(param_iss, value, ','))
  //       return base::kStatusCodeErrorInvalidValue;
  //     // reserved_param_[key] = base::Value::fromString(value);
  //   }

  //   // 读取 reserved_
  //   if (!std::getline(iss, token, ':'))
  //     return base::kStatusCodeErrorInvalidValue;
  //   if (token != "reserved_") return base::kStatusCodeErrorInvalidValue;
  //   if (!std::getline(iss, token, ','))
  //     return base::kStatusCodeErrorInvalidValue;
  //   reserved_ = std::stoull(token);

  //   return base::kStatusCodeOk;
  // }

 public:
  // 保留字段,key-value的形式
  std::map<std::string, base::Value> reserved_param_;
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
    if (std::getline(iss, token, ':')) {
      if (token == "epsilon_") {
        if (!std::getline(iss, token, ','))
          return base::kStatusCodeErrorInvalidValue;
        epsilon_ = std::stof(token);
      }
    }

    // 读取 momentum_
    if (std::getline(iss, token, ':')) {
      if (token == "momentum_") {
        if (!std::getline(iss, token, ','))
          return base::kStatusCodeErrorInvalidValue;
        momentum_ = std::stof(token);
      }
    }

    // 读取 training_mode_
    if (std::getline(iss, token, ':')) {
      if (token != "training_mode_") {
        if (!std::getline(iss, token, ','))
          return base::kStatusCodeErrorInvalidValue;
        training_mode_ = std::stoi(token);
      }
    }

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

    NNDEPLOY_LOGE("hello world\n");

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

    NNDEPLOY_LOGE("hello world\n");

    // 读取 group_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "group_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    group_ = std::stoi(token);

    NNDEPLOY_LOGE("hello world\n");

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
    NNDEPLOY_LOGE("hello world\n");

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

    NNDEPLOY_LOGE("hello world\n");

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

    NNDEPLOY_LOGE("hello world\n");

    // 读取 is_fusion_op_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "is_fusion_op_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    is_fusion_op_ = (token == "1");

    NNDEPLOY_LOGE("hello world\n");

    // 读取 activate_op_
    if (!std::getline(iss, token, ':'))
      return base::kStatusCodeErrorInvalidValue;
    if (token != "activate_op_") return base::kStatusCodeErrorInvalidValue;
    if (!std::getline(iss, token, ','))
      return base::kStatusCodeErrorInvalidValue;
    activate_op_ = stringToOpType(token);

    NNDEPLOY_LOGE("hello world\n");

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

 public:
  float eps_ = 1e-6;
  bool is_last_ = false;
};

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_IR_H_ */