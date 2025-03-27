#include "nndeploy/ir/op_param.h"

#include <pybind11/stl.h>

#include <vector>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace ir {

class PyOpParamCreator : public OpParamCreator {
 public:
  using OpParamCreator::OpParamCreator;

  // 重写纯虚函数
  std::shared_ptr<base::Param> createOpParam(OpType type) override {
    PYBIND11_OVERLOAD_PURE_NAME(std::shared_ptr<base::Param>,  // 返回类型
                                OpParamCreator,                // 父类
                                "create_op_param",             // 函数名
                                createOpParam,                 // 函数名
                                type                           // 参数
    );
  }
};

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  // export as ir.OpType
  py::enum_<OpType>(m, "OpType")
      .value("Net", kOpTypeNet)
      .value("Abs", kOpTypeAbs)
      .value("Add", kOpTypeAdd)
      .value("Acos", kOpTypeAcos)
      .value("Adam", kOpTypeAdam)
      .value("And", kOpTypeAnd)
      .value("ArgMax", kOpTypeArgMax)
      .value("ArgMin", kOpTypeArgMin)
      .value("Asin", kOpTypeAsin)
      .value("Atan", kOpTypeAtan)
      .value("AveragePool", kOpTypeAveragePool)
      .value("BatchNormalization", kOpTypeBatchNormalization)
      .value("Cast", kOpTypeCast)
      .value("Ceil", kOpTypeCeil)
      .value("Clip", kOpTypeClip)
      .value("Concat", kOpTypeConcat)
      .value("Constant", kOpTypeConstant)
      .value("Conv", kOpTypeConv)
      .value("Cos", kOpTypeCos)
      .value("Cosh", kOpTypeCosh)
      .value("DepthToSpace", kOpTypeDepthToSpace)
      .value("DequantizeLinear", kOpTypeDequantizeLinear)
      .value("Det", kOpTypeDet)
      .value("Div", kOpTypeDiv)
      .value("Dropout", kOpTypeDropout)
      .value("Einsum", kOpTypeEinsum)
      .value("Elu", kOpTypeElu)
      .value("Equal", kOpTypeEqual)
      .value("Erf", kOpTypeErf)
      .value("Exp", kOpTypeExp)
      .value("Flatten", kOpTypeFlatten)
      .value("Floor", kOpTypeFloor)
      .value("Gather", kOpTypeGather)
      .value("Gemm", kOpTypeGemm)
      .value("GlobalAveragePool", kOpTypeGlobalAveragePool)
      .value("GlobalLpPool", kOpTypeGlobalLpPool)
      .value("GlobalMaxPool", kOpTypeGlobalMaxPool)
      .value("Greater", kOpTypeGreater)
      .value("HardSigmoid", kOpTypeHardSigmoid)
      .value("Identity", kOpTypeIdentity)
      .value("If", kOpTypeIf)
      .value("ImageScaler", kOpTypeImageScaler)
      .value("InstanceNormalization", kOpTypeInstanceNormalization)
      .value("IsInf", kOpTypeIsInf)
      .value("IsNaN", kOpTypeIsNaN)
      .value("LRN", kOpTypeLRN)
      .value("LSTM", kOpTypeLSTM)
      .value("LeakyRelu", kOpTypeLeakyRelu)
      .value("Less", kOpTypeLess)
      .value("Log", kOpTypeLog)
      .value("LogSoftmax", kOpTypeLogSoftmax)
      .value("Loop", kOpTypeLoop)
      .value("LpNormalization", kOpTypeLpNormalization)
      .value("LpPool", kOpTypeLpPool)
      .value("MatMul", kOpTypeMatMul)
      .value("MatMulInteger", kOpTypeMatMulInteger)
      .value("Max", kOpTypeMax)
      .value("MaxPool", kOpTypeMaxPool)
      .value("MaxRoiPool", kOpTypeMaxRoiPool)
      .value("MaxUnpool", kOpTypeMaxUnpool)
      .value("Mean", kOpTypeMean)
      .value("Min", kOpTypeMin)
      .value("Mod", kOpTypeMod)
      .value("Momentum", kOpTypeMomentum)
      .value("Mul", kOpTypeMul)
      .value("Multinomial", kOpTypeMultinomial)
      .value("Neg", kOpTypeNeg)
      .value("NegLogSoftmax", kOpTypeNegLogSoftmax)
      .value("NonMaxSuppression", kOpTypeNonMaxSuppression)
      .value("NonZero", kOpTypeNonZero)
      .value("Not", kOpTypeNot)
      .value("OneHot", kOpTypeOneHot)
      .value("OnesLike", kOpTypeOnesLike)
      .value("Or", kOpTypeOr)
      .value("Pad", kOpTypePad)
      .value("Pow", kOpTypePow)
      .value("PRelu", kOpTypePRelu)
      .value("QLinearConv", kOpTypeQLinearConv)
      .value("QLinearMatMul", kOpTypeQLinearMatMul)
      .value("QuantizeLinear", kOpTypeQuantizeLinear)
      .value("RNN", kOpTypeRNN)
      .value("RandomNormal", kOpTypeRandomNormal)
      .value("RandomNormalLike", kOpTypeRandomNormalLike)
      .value("RandomUniform", kOpTypeRandomUniform)
      .value("RandomUniformLike", kOpTypeRandomUniformLike)
      .value("Range", kOpTypeRange)
      .value("Reciprocal", kOpTypeReciprocal)
      .value("ReduceL1", kOpTypeReduceL1)
      .value("ReduceL2", kOpTypeReduceL2)
      .value("ReduceLogSum", kOpTypeReduceLogSum)
      .value("ReduceLogSumExp", kOpTypeReduceLogSumExp)
      .value("ReduceMax", kOpTypeReduceMax)
      .value("ReduceMean", kOpTypeReduceMean)
      .value("ReduceMin", kOpTypeReduceMin)
      .value("ReduceProd", kOpTypeReduceProd)
      .value("ReduceSum", kOpTypeReduceSum)
      .value("ReduceSumSquare", kOpTypeReduceSumSquare)
      .value("Relu", kOpTypeRelu)
      .value("Reshape", kOpTypeReshape)
      .value("Resize", kOpTypeResize)
      .value("ReverseSequence", kOpTypeReverseSequence)
      .value("RoiAlign", kOpTypeRoiAlign)
      .value("Round", kOpTypeRound)
      .value("Scale", kOpTypeScale)
      .value("Scan", kOpTypeScan)
      .value("Scatter", kOpTypeScatter)
      .value("Selu", kOpTypeSelu)
      .value("SequenceAt", kOpTypeSequenceAt)
      .value("SequenceConstruct", kOpTypeSequenceConstruct)
      .value("SequenceEmpty", kOpTypeSequenceEmpty)
      .value("SequenceErase", kOpTypeSequenceErase)
      .value("SequenceInsert", kOpTypeSequenceInsert)
      .value("SequenceLength", kOpTypeSequenceLength)
      .value("Shape", kOpTypeShape)
      .value("Shrink", kOpTypeShrink)
      .value("Sigmoid", kOpTypeSigmoid)
      .value("Sign", kOpTypeSign)
      .value("Sin", kOpTypeSin)
      .value("Sinh", kOpTypeSinh)
      .value("Size", kOpTypeSize)
      .value("Slice", kOpTypeSlice)
      .value("Softmax", kOpTypeSoftmax)
      .value("Softplus", kOpTypeSoftplus)
      .value("Softsign", kOpTypeSoftsign)
      .value("SpaceToDepth", kOpTypeSpaceToDepth)
      .value("Split", kOpTypeSplit)
      .value("Sqrt", kOpTypeSqrt)
      .value("Squeeze", kOpTypeSqueeze)
      .value("Sub", kOpTypeSub)
      .value("Sum", kOpTypeSum)
      .value("Tan", kOpTypeTan)
      .value("Tanh", kOpTypeTanh)
      .value("TfIdf", kOpTypeTfIdf)
      .value("ThresholdedRelu", kOpTypeThresholdedRelu)
      .value("Tile", kOpTypeTile)
      .value("TopK", kOpTypeTopK)
      .value("Transpose", kOpTypeTranspose)
      .value("Unsqueeze", kOpTypeUnsqueeze)
      .value("Upsample", kOpTypeUpsample)
      .value("Where", kOpTypeWhere)
      .value("Xor", kOpTypeXor)
      .value("RMSNorm", kOpTypeRMSNorm)
      .value("Embedding", kOpTypeEmbedding)
      .value("kOpTypeNone", kOpTypeNone);

  m.def("op_type_to_string", &opTypeToString);
  m.def("string_to_op_type", &stringToOpType);

  // 导出 OpParamCreator 类
  py::class_<OpParamCreator, PyOpParamCreator, std::shared_ptr<OpParamCreator>>(
      m, "OpParamCreator")
      .def(py::init<>())
      .def("create_op_param", &OpParamCreator::createOpParam);

  // export register_op_param_creator
  m.def("register_op_param_creator",
        [](OpType type, std::shared_ptr<OpParamCreator> creator) {
          getGlobalOpParamCreatorMap()[type] = creator;
          // for (auto &item : getGlobalOpParamCreatorMap()) {
          //   std::cout << opTypeToString(item.first) << std::endl;
          // }
        });

  // export create_op_param
  m.def("create_op_param", [](OpType type) {
    auto creator = getGlobalOpParamCreatorMap()[type];
    if (creator) {
      return creator->createOpParam(type);
    } else {
      throw std::runtime_error("No OpParamCreator registered for OpType " +
                               opTypeToString(type));
    }
  });

  // 导出 OpParam 类
  py::class_<OpParam, base::Param, std::shared_ptr<OpParam>>(m, "OpParam",
                                                             py::dynamic_attr())
      .def(py::init<>());

  // 导出 BatchNormalizationParam 类
  py::class_<BatchNormalizationParam, OpParam,
             std::shared_ptr<BatchNormalizationParam>>(
      m, "BatchNormalizationParam")
      .def(py::init<>())
      .def_readwrite("epsilon_", &BatchNormalizationParam::epsilon_)
      .def_readwrite("momentum_", &BatchNormalizationParam::momentum_)
      .def_readwrite("training_mode_",
                     &BatchNormalizationParam::training_mode_);

  // 导出 ConcatParam 类
  py::class_<ConcatParam, OpParam, std::shared_ptr<ConcatParam>>(m,
                                                                 "ConcatParam")
      .def(py::init<>())
      .def_readwrite("axis_", &ConcatParam::axis_);

  // 导出 ConvParam 类
  py::class_<ConvParam, OpParam, std::shared_ptr<ConvParam>>(m, "ConvParam")
      .def(py::init<>())
      .def_readwrite("auto_pad_", &ConvParam::auto_pad_)
      .def_readwrite("dilations_", &ConvParam::dilations_)
      .def_readwrite("group_", &ConvParam::group_)
      .def_readwrite("kernel_shape_", &ConvParam::kernel_shape_)
      .def_readwrite("pads_", &ConvParam::pads_)
      .def_readwrite("strides_", &ConvParam::strides_);

  // 导出 MaxPoolParam 类
  py::class_<MaxPoolParam, OpParam, std::shared_ptr<MaxPoolParam>>(
      m, "MaxPoolParam")
      .def(py::init<>())
      .def_readwrite("auto_pad_", &MaxPoolParam::auto_pad_)
      .def_readwrite("ceil_mode_", &MaxPoolParam::ceil_mode_)
      .def_readwrite("dilations_", &MaxPoolParam::dilations_)
      .def_readwrite("kernel_shape_", &MaxPoolParam::kernel_shape_)
      .def_readwrite("pads_", &MaxPoolParam::pads_)
      .def_readwrite("storage_order_", &MaxPoolParam::storage_order_)
      .def_readwrite("strides_", &MaxPoolParam::strides_);

  // 导出 ReshapeParam 类
  py::class_<ReshapeParam, OpParam, std::shared_ptr<ReshapeParam>>(
      m, "ReshapeParam")
      .def(py::init<>())
      .def_readwrite("allowzero_", &ReshapeParam::allowzero_);

  // 导出 ResizeParam 类
  py::class_<ResizeParam, OpParam, std::shared_ptr<ResizeParam>>(m,
                                                                 "ResizeParam")
      .def(py::init<>())
      .def_readwrite("antialias_", &ResizeParam::antialias_)
      .def_readwrite("axes_", &ResizeParam::axes_)
      .def_readwrite("coordinate_transformation_mode_",
                     &ResizeParam::coordinate_transformation_mode_)
      .def_readwrite("cubic_coeff_a_", &ResizeParam::cubic_coeff_a_)
      .def_readwrite("exclude_outside_", &ResizeParam::exclude_outside_)
      .def_readwrite("extrapolation_value_", &ResizeParam::extrapolation_value_)
      .def_readwrite("keep_aspect_ratio_policy_",
                     &ResizeParam::keep_aspect_ratio_policy_)
      .def_readwrite("mode_", &ResizeParam::mode_)
      .def_readwrite("nearest_mode_", &ResizeParam::nearest_mode_);

  // 导出 SoftmaxParam 类
  py::class_<SoftmaxParam, OpParam, std::shared_ptr<SoftmaxParam>>(
      m, "SoftmaxParam")
      .def(py::init<>())
      .def_readwrite("axis_", &SoftmaxParam::axis_);

  // 导出 SplitParam 类
  py::class_<SplitParam, OpParam, std::shared_ptr<SplitParam>>(m, "SplitParam")
      .def(py::init<>())
      .def_readwrite("axis_", &SplitParam::axis_)
      .def_readwrite("num_outputs_", &SplitParam::num_outputs_);

  // 导出 TransposeParam 类
  py::class_<TransposeParam, OpParam, std::shared_ptr<TransposeParam>>(
      m, "TransposeParam")
      .def(py::init<>())
      .def_readwrite("perm_", &TransposeParam::perm_);

  // 导出 RMSNormParam 类
  py::class_<RMSNormParam, OpParam, std::shared_ptr<RMSNormParam>>(
      m, "RMSNormParam")
      .def(py::init<>())
      .def_readwrite("eps_", &RMSNormParam::eps_)
      .def_readwrite("is_last_", &RMSNormParam::is_last_);

  py::class_<FlattenParam, OpParam, std::shared_ptr<FlattenParam>>(
      m, "FlattenParam")
      .def(py::init<>())
      .def_readwrite("axis_", &FlattenParam::axis_);

  py::class_<GemmParam, OpParam, std::shared_ptr<GemmParam>>(m, "GemmParam")
      .def(py::init<>())
      .def_readwrite("alpha_", &GemmParam::alpha_)
      .def_readwrite("beta_", &GemmParam::beta_)
      .def_readwrite("trans_a_", &GemmParam::trans_a_)
      .def_readwrite("trans_b_", &GemmParam::trans_b_);

  py::class_<QuantizeLinearParam, OpParam,
             std::shared_ptr<QuantizeLinearParam>>(m, "QuantizeLinearParam")
      .def(py::init<>())
      .def_readwrite("axis_", &QuantizeLinearParam::axis_)
      .def_readwrite("saturate_", &QuantizeLinearParam::saturate_);

  py::class_<DequantizeLinearParam, OpParam,
             std::shared_ptr<DequantizeLinearParam>>(m, "DequantizeLinearParam")
      .def(py::init<>())
      .def_readwrite("axis_", &DequantizeLinearParam::axis_);

  py::class_<QLinearConvParam, OpParam, std::shared_ptr<QLinearConvParam>>(
      m, "QLinearConvParam")
      .def(py::init<>())
      .def_readwrite("auto_pad_", &QLinearConvParam::auto_pad_)
      .def_readwrite("dilations_", &QLinearConvParam::dilations_)
      .def_readwrite("group_", &QLinearConvParam::group_)
      .def_readwrite("kernel_shape_", &QLinearConvParam::kernel_shape_)
      .def_readwrite("pads_", &QLinearConvParam::pads_)
      .def_readwrite("strides_", &QLinearConvParam::strides_);
}
}  // namespace ir
}  // namespace nndeploy