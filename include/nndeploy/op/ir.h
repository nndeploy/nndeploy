
#ifndef _NNDEPLOY_OP_IR_H_
#define _NNDEPLOY_OP_IR_H_

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
namespace op {
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
 * deeolink的分类
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

  kOpTypeNone,
};

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

 public:
  // 描述模型的名称
  std::string name_;
  // 模型算子列表
  std::vector<std::shared_ptr<OpDesc>> op_descs_;
  // 模型权重
  std::map<std::string, device::Tensor *> weights_;
  // 模型输入
  std::vector<std::shared_ptr<ValueDesc>> inputs_;
  // 模型输出
  std::vector<std::shared_ptr<ValueDesc>> outputs_;
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
  virtual ~OpParamCreator(){};
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

class OpParam : public base::Param {
 public:
  OpParam(){};
  virtual ~OpParam(){};

  PARAM_COPY(OpParam)
  PARAM_COPY_TO(OpParam)
  // 保留字段,key-value的形式
  std::map<std::string, base::Value> reserved_param_;
  // 保留字段,也可以充void *使用
  size_t reserved_;
};

class ConcatParam : public OpParam {
 public:
  ConcatParam(){};
  virtual ~ConcatParam(){};

  PARAM_COPY(ConcatParam)
  PARAM_COPY_TO(ConcatParam)
  int axis_;
};

class ConvParam : public OpParam {
 public:
  // 构造函数
  ConvParam() {}
  virtual ~ConvParam() {}

  // 复制构造函数
  ConvParam(const ConvParam &other) {
    strides_ = other.strides_;
    auto_pad_ = other.auto_pad_;
    dilations_ = other.dilations_;
    kernel_shape_ = other.kernel_shape_;
    pads_ = other.pads_;
    group_ = other.group_;
  }

  // 赋值运算符
  ConvParam &operator=(const ConvParam &other) {
    if (this != &other) {
      strides_ = other.strides_;
      auto_pad_ = other.auto_pad_;
      dilations_ = other.dilations_;
      kernel_shape_ = other.kernel_shape_;
      pads_ = other.pads_;
      group_ = other.group_;
    }
    return *this;
  }

  PARAM_COPY(ConvParam)
  PARAM_COPY_TO(ConvParam)

  // 自动填充方式
  std::string auto_pad_ = "";
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

  bool is_fusion_op_ = false;
  OpType activate_op_ = kOpTypeRelu;
};
// MaxPool 参数类
class MaxPoolParam : public OpParam {
 public:
  MaxPoolParam() {}
  virtual ~MaxPoolParam() {}

  MaxPoolParam(const MaxPoolParam &other) {
    strides_ = other.strides_;
    kernel_shape_ = other.kernel_shape_;
    pads_ = other.pads_;
  }

  MaxPoolParam &operator=(const MaxPoolParam &other) {
    if (this != &other) {
      strides_ = other.strides_;
      kernel_shape_ = other.kernel_shape_;
      pads_ = other.pads_;
    }
    return *this;
  }

  PARAM_COPY(MaxPoolParam)
  PARAM_COPY_TO(MaxPoolParam)
  std::string auto_pad_ = "";             // 自动填充方式
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
  ReshapeParam() {}
  virtual ~ReshapeParam() {}

  ReshapeParam(const ReshapeParam &other) { allowzero_ = other.allowzero_; }

  ReshapeParam &operator=(const ReshapeParam &other) {
    if (this != &other) {
      allowzero_ = other.allowzero_;
    }
    return *this;
  }

  PARAM_COPY(ReshapeParam)
  PARAM_COPY_TO(ReshapeParam)

  int allowzero_;  // 是否允许0
};

// Resize 参数类
class ResizeParam : public OpParam {
 public:
  ResizeParam() {}
  virtual ~ResizeParam() {}

  PARAM_COPY(ResizeParam)
  PARAM_COPY_TO(ResizeParam)

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
  SoftmaxParam() : axis_(1) {}  // 默认轴为1
  virtual ~SoftmaxParam() {}

  PARAM_COPY(SoftmaxParam)
  PARAM_COPY_TO(SoftmaxParam)

  int axis_;  // 应用 Softmax 的轴
};

// Split 参数类
class SplitParam : public OpParam {
 public:
  SplitParam() : OpParam() {}  // 默认轴为0，分割数为1
  virtual ~SplitParam() {}

  PARAM_COPY(SplitParam)
  PARAM_COPY_TO(SplitParam)

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

  std::vector<int> perm_;
};

// TODO: @Leonisux:
// 补充llama的算子的参数

}  // namespace op
}  // namespace nndeploy

#endif /* _NNDEPLOY_OP_IR_H_ */