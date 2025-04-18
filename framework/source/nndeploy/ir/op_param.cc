
#include "nndeploy/ir/op_param.h"

namespace nndeploy {
namespace ir {

static const std::map<OpType, std::string> g_optype_string_map = {
    {kOpTypeNet, "kOpTypeNet"},
    {kOpTypeAbs, "kOpTypeAbs"},
    {kOpTypeAdd, "kOpTypeAdd"},
    {kOpTypeAcos, "kOpTypeAcos"},
    {kOpTypeAdam, "kOpTypeAdam"},
    {kOpTypeAnd, "kOpTypeAnd"},
    {kOpTypeArgMax, "kOpTypeArgMax"},
    {kOpTypeArgMin, "kOpTypeArgMin"},
    {kOpTypeAsin, "kOpTypeAsin"},
    {kOpTypeAtan, "kOpTypeAtan"},
    {kOpTypeAveragePool, "kOpTypeAveragePool"},
    {kOpTypeBatchNormalization, "kOpTypeBatchNormalization"},
    {kOpTypeCast, "kOpTypeCast"},
    {kOpTypeCeil, "kOpTypeCeil"},
    {kOpTypeClip, "kOpTypeClip"},
    {kOpTypeConcat, "kOpTypeConcat"},
    {kOpTypeConstant, "kOpTypeConstant"},
    {kOpTypeConv, "kOpTypeConv"},
    {kOpTypeCos, "kOpTypeCos"},
    {kOpTypeCosh, "kOpTypeCosh"},
    {kOpTypeDepthToSpace, "kOpTypeDepthToSpace"},
    {kOpTypeDequantizeLinear, "kOpTypeDequantizeLinear"},
    {kOpTypeDet, "kOpTypeDet"},
    {kOpTypeDiv, "kOpTypeDiv"},
    {kOpTypeDropout, "kOpTypeDropout"},
    {kOpTypeEinsum, "kOpTypeEinsum"},
    {kOpTypeElu, "kOpTypeElu"},
    {kOpTypeEqual, "kOpTypeEqual"},
    {kOpTypeErf, "kOpTypeErf"},
    {kOpTypeExp, "kOpTypeExp"},
    {kOpTypeFlatten, "kOpTypeFlatten"},
    {kOpTypeFloor, "kOpTypeFloor"},
    {kOpTypeGather, "kOpTypeGather"},
    {kOpTypeGemm, "kOpTypeGemm"},
    {kOpTypeGlobalAveragePool, "kOpTypeGlobalAveragePool"},
    {kOpTypeGlobalLpPool, "kOpTypeGlobalLpPool"},
    {kOpTypeGlobalMaxPool, "kOpTypeGlobalMaxPool"},
    {kOpTypeGreater, "kOpTypeGreater"},
    {kOpTypeHardSigmoid, "kOpTypeHardSigmoid"},
    {kOpTypeIdentity, "kOpTypeIdentity"},
    {kOpTypeIf, "kOpTypeIf"},
    {kOpTypeImageScaler, "kOpTypeImageScaler"},
    {kOpTypeInstanceNormalization, "kOpTypeInstanceNormalization"},
    {kOpTypeIsInf, "kOpTypeIsInf"},
    {kOpTypeIsNaN, "kOpTypeIsNaN"},
    {kOpTypeLRN, "kOpTypeLRN"},
    {kOpTypeLSTM, "kOpTypeLSTM"},
    {kOpTypeLeakyRelu, "kOpTypeLeakyRelu"},
    {kOpTypeLess, "kOpTypeLess"},
    {kOpTypeLog, "kOpTypeLog"},
    {kOpTypeLogSoftmax, "kOpTypeLogSoftmax"},
    {kOpTypeLoop, "kOpTypeLoop"},
    {kOpTypeLpNormalization, "kOpTypeLpNormalization"},
    {kOpTypeLpPool, "kOpTypeLpPool"},
    {kOpTypeMatMul, "kOpTypeMatMul"},
    {kOpTypeMatMulInteger, "kOpTypeMatMulInteger"},
    {kOpTypeMax, "kOpTypeMax"},
    {kOpTypeMaxPool, "kOpTypeMaxPool"},
    {kOpTypeMaxRoiPool, "kOpTypeMaxRoiPool"},
    {kOpTypeMaxUnpool, "kOpTypeMaxUnpool"},
    {kOpTypeMean, "kOpTypeMean"},
    {kOpTypeMin, "kOpTypeMin"},
    {kOpTypeMod, "kOpTypeMod"},
    {kOpTypeMomentum, "kOpTypeMomentum"},
    {kOpTypeMul, "kOpTypeMul"},
    {kOpTypeMultinomial, "kOpTypeMultinomial"},
    {kOpTypeNeg, "kOpTypeNeg"},
    {kOpTypeNegLogSoftmax, "kOpTypeNegLogSoftmax"},
    {kOpTypeNonMaxSuppression, "kOpTypeNonMaxSuppression"},
    {kOpTypeNonZero, "kOpTypeNonZero"},
    {kOpTypeNot, "kOpTypeNot"},
    {kOpTypeOneHot, "kOpTypeOneHot"},
    {kOpTypeOnesLike, "kOpTypeOnesLike"},
    {kOpTypeOr, "kOpTypeOr"},
    {kOpTypePad, "kOpTypePad"},
    {kOpTypePow, "kOpTypePow"},
    {kOpTypePRelu, "kOpTypePRelu"},
    {kOpTypeQLinearConv, "kOpTypeQLinearConv"},
    {kOpTypeQLinearMatMul, "kOpTypeQLinearMatMul"},
    {kOpTypeQuantizeLinear, "kOpTypeQuantizeLinear"},
    {kOpTypeRNN, "kOpTypeRNN"},
    {kOpTypeRandomNormal, "kOpTypeRandomNormal"},
    {kOpTypeRandomNormalLike, "kOpTypeRandomNormalLike"},
    {kOpTypeRandomUniform, "kOpTypeRandomUniform"},
    {kOpTypeRandomUniformLike, "kOpTypeRandomUniformLike"},
    {kOpTypeRange, "kOpTypeRange"},
    {kOpTypeReciprocal, "kOpTypeReciprocal"},
    {kOpTypeReduceL1, "kOpTypeReduceL1"},
    {kOpTypeReduceL2, "kOpTypeReduceL2"},
    {kOpTypeReduceLogSum, "kOpTypeReduceLogSum"},
    {kOpTypeReduceLogSumExp, "kOpTypeReduceLogSumExp"},
    {kOpTypeReduceMax, "kOpTypeReduceMax"},
    {kOpTypeReduceMean, "kOpTypeReduceMean"},
    {kOpTypeReduceMin, "kOpTypeReduceMin"},
    {kOpTypeReduceProd, "kOpTypeReduceProd"},
    {kOpTypeReduceSum, "kOpTypeReduceSum"},
    {kOpTypeReduceSumSquare, "kOpTypeReduceSumSquare"},
    {kOpTypeRelu, "kOpTypeRelu"},
    {kOpTypeReshape, "kOpTypeReshape"},
    {kOpTypeResize, "kOpTypeResize"},
    {kOpTypeReverseSequence, "kOpTypeReverseSequence"},
    {kOpTypeRoiAlign, "kOpTypeRoiAlign"},
    {kOpTypeRound, "kOpTypeRound"},
    {kOpTypeScale, "kOpTypeScale"},
    {kOpTypeScan, "kOpTypeScan"},
    {kOpTypeScatter, "kOpTypeScatter"},
    {kOpTypeSelu, "kOpTypeSelu"},
    {kOpTypeSequenceAt, "kOpTypeSequenceAt"},
    {kOpTypeSequenceConstruct, "kOpTypeSequenceConstruct"},
    {kOpTypeSequenceEmpty, "kOpTypeSequenceEmpty"},
    {kOpTypeSequenceErase, "kOpTypeSequenceErase"},
    {kOpTypeSequenceInsert, "kOpTypeSequenceInsert"},
    {kOpTypeSequenceLength, "kOpTypeSequenceLength"},
    {kOpTypeShape, "kOpTypeShape"},
    {kOpTypeShrink, "kOpTypeShrink"},
    {kOpTypeSigmoid, "kOpTypeSigmoid"},
    {kOpTypeSign, "kOpTypeSign"},
    {kOpTypeSin, "kOpTypeSin"},
    {kOpTypeSinh, "kOpTypeSinh"},
    {kOpTypeSize, "kOpTypeSize"},
    {kOpTypeSlice, "kOpTypeSlice"},
    {kOpTypeSoftmax, "kOpTypeSoftmax"},
    {kOpTypeSoftplus, "kOpTypeSoftplus"},
    {kOpTypeSoftsign, "kOpTypeSoftsign"},
    {kOpTypeSpaceToDepth, "kOpTypeSpaceToDepth"},
    {kOpTypeSplit, "kOpTypeSplit"},
    {kOpTypeSqrt, "kOpTypeSqrt"},
    {kOpTypeSqueeze, "kOpTypeSqueeze"},
    {kOpTypeSub, "kOpTypeSub"},
    {kOpTypeSum, "kOpTypeSum"},
    {kOpTypeTan, "kOpTypeTan"},
    {kOpTypeTanh, "kOpTypeTanh"},
    {kOpTypeTfIdf, "kOpTypeTfIdf"},
    {kOpTypeThresholdedRelu, "kOpTypeThresholdedRelu"},
    {kOpTypeTile, "kOpTypeTile"},
    {kOpTypeTopK, "kOpTypeTopK"},
    {kOpTypeTranspose, "kOpTypeTranspose"},
    {kOpTypeUnsqueeze, "kOpTypeUnsqueeze"},
    {kOpTypeUpsample, "kOpTypeUpsample"},
    {kOpTypeWhere, "kOpTypeWhere"},
    {kOpTypeXor, "kOpTypeXor"},
    {kOpTypeRMSNorm, "kOpTypeRMSNorm"},
    {kOpTypeEmbedding, "kOpTypeEmbedding"},

    {kOpTypeNone, "kOpTypeNone"},
};

static const std::map<std::string, OpType> g_string_optype_map = {
    {"kOpTypeNet", kOpTypeNet},
    {"kOpTypeAbs", kOpTypeAbs},
    {"kOpTypeAdd", kOpTypeAdd},
    {"kOpTypeAcos", kOpTypeAcos},
    {"kOpTypeAdam", kOpTypeAdam},
    {"kOpTypeAnd", kOpTypeAnd},
    {"kOpTypeArgMax", kOpTypeArgMax},
    {"kOpTypeArgMin", kOpTypeArgMin},
    {"kOpTypeAsin", kOpTypeAsin},
    {"kOpTypeAtan", kOpTypeAtan},
    {"kOpTypeAveragePool", kOpTypeAveragePool},
    {"kOpTypeBatchNormalization", kOpTypeBatchNormalization},
    {"kOpTypeCast", kOpTypeCast},
    {"kOpTypeCeil", kOpTypeCeil},
    {"kOpTypeClip", kOpTypeClip},
    {"kOpTypeConcat", kOpTypeConcat},
    {"kOpTypeConstant", kOpTypeConstant},
    {"kOpTypeConv", kOpTypeConv},
    {"kOpTypeCos", kOpTypeCos},
    {"kOpTypeCosh", kOpTypeCosh},
    {"kOpTypeDepthToSpace", kOpTypeDepthToSpace},
    {"kOpTypeDequantizeLinear", kOpTypeDequantizeLinear},
    {"kOpTypeDet", kOpTypeDet},
    {"kOpTypeDiv", kOpTypeDiv},
    {"kOpTypeDropout", kOpTypeDropout},
    {"kOpTypeEinsum", kOpTypeEinsum},
    {"kOpTypeElu", kOpTypeElu},
    {"kOpTypeEqual", kOpTypeEqual},
    {"kOpTypeErf", kOpTypeErf},
    {"kOpTypeExp", kOpTypeExp},
    {"kOpTypeFlatten", kOpTypeFlatten},
    {"kOpTypeFloor", kOpTypeFloor},
    {"kOpTypeGather", kOpTypeGather},
    {"kOpTypeGemm", kOpTypeGemm},
    {"kOpTypeGlobalAveragePool", kOpTypeGlobalAveragePool},
    {"kOpTypeGlobalLpPool", kOpTypeGlobalLpPool},
    {"kOpTypeGlobalMaxPool", kOpTypeGlobalMaxPool},
    {"kOpTypeGreater", kOpTypeGreater},
    {"kOpTypeHardSigmoid", kOpTypeHardSigmoid},
    {"kOpTypeIdentity", kOpTypeIdentity},
    {"kOpTypeIf", kOpTypeIf},
    {"kOpTypeImageScaler", kOpTypeImageScaler},
    {"kOpTypeInstanceNormalization", kOpTypeInstanceNormalization},
    {"kOpTypeIsInf", kOpTypeIsInf},
    {"kOpTypeIsNaN", kOpTypeIsNaN},
    {"kOpTypeLRN", kOpTypeLRN},
    {"kOpTypeLSTM", kOpTypeLSTM},
    {"kOpTypeLeakyRelu", kOpTypeLeakyRelu},
    {"kOpTypeLess", kOpTypeLess},
    {"kOpTypeLog", kOpTypeLog},
    {"kOpTypeLogSoftmax", kOpTypeLogSoftmax},
    {"kOpTypeLoop", kOpTypeLoop},
    {"kOpTypeLpNormalization", kOpTypeLpNormalization},
    {"kOpTypeLpPool", kOpTypeLpPool},
    {"kOpTypeMatMul", kOpTypeMatMul},
    {"kOpTypeMatMulInteger", kOpTypeMatMulInteger},
    {"kOpTypeMax", kOpTypeMax},
    {"kOpTypeMaxPool", kOpTypeMaxPool},
    {"kOpTypeMaxRoiPool", kOpTypeMaxRoiPool},
    {"kOpTypeMaxUnpool", kOpTypeMaxUnpool},
    {"kOpTypeMean", kOpTypeMean},
    {"kOpTypeMin", kOpTypeMin},
    {"kOpTypeMod", kOpTypeMod},
    {"kOpTypeMomentum", kOpTypeMomentum},
    {"kOpTypeMul", kOpTypeMul},
    {"kOpTypeMultinomial", kOpTypeMultinomial},
    {"kOpTypeNeg", kOpTypeNeg},
    {"kOpTypeNegLogSoftmax", kOpTypeNegLogSoftmax},
    {"kOpTypeNonMaxSuppression", kOpTypeNonMaxSuppression},
    {"kOpTypeNonZero", kOpTypeNonZero},
    {"kOpTypeNot", kOpTypeNot},
    {"kOpTypeOneHot", kOpTypeOneHot},
    {"kOpTypeOnesLike", kOpTypeOnesLike},
    {"kOpTypeOr", kOpTypeOr},
    {"kOpTypePad", kOpTypePad},
    {"kOpTypePow", kOpTypePow},
    {"kOpTypePRelu", kOpTypePRelu},
    {"kOpTypeQLinearConv", kOpTypeQLinearConv},
    {"kOpTypeQLinearMatMul", kOpTypeQLinearMatMul},
    {"kOpTypeQuantizeLinear", kOpTypeQuantizeLinear},
    {"kOpTypeRNN", kOpTypeRNN},
    {"kOpTypeRandomNormal", kOpTypeRandomNormal},
    {"kOpTypeRandomNormalLike", kOpTypeRandomNormalLike},
    {"kOpTypeRandomUniform", kOpTypeRandomUniform},
    {"kOpTypeRandomUniformLike", kOpTypeRandomUniformLike},
    {"kOpTypeRange", kOpTypeRange},
    {"kOpTypeReciprocal", kOpTypeReciprocal},
    {"kOpTypeReduceL1", kOpTypeReduceL1},
    {"kOpTypeReduceL2", kOpTypeReduceL2},
    {"kOpTypeReduceLogSum", kOpTypeReduceLogSum},
    {"kOpTypeReduceLogSumExp", kOpTypeReduceLogSumExp},
    {"kOpTypeReduceMax", kOpTypeReduceMax},
    {"kOpTypeReduceMean", kOpTypeReduceMean},
    {"kOpTypeReduceMin", kOpTypeReduceMin},
    {"kOpTypeReduceProd", kOpTypeReduceProd},
    {"kOpTypeReduceSum", kOpTypeReduceSum},
    {"kOpTypeReduceSumSquare", kOpTypeReduceSumSquare},
    {"kOpTypeRelu", kOpTypeRelu},
    {"kOpTypeReshape", kOpTypeReshape},
    {"kOpTypeResize", kOpTypeResize},
    {"kOpTypeReverseSequence", kOpTypeReverseSequence},
    {"kOpTypeRoiAlign", kOpTypeRoiAlign},
    {"kOpTypeRound", kOpTypeRound},
    {"kOpTypeScale", kOpTypeScale},
    {"kOpTypeScan", kOpTypeScan},
    {"kOpTypeScatter", kOpTypeScatter},
    {"kOpTypeSelu", kOpTypeSelu},
    {"kOpTypeSequenceAt", kOpTypeSequenceAt},
    {"kOpTypeSequenceConstruct", kOpTypeSequenceConstruct},
    {"kOpTypeSequenceEmpty", kOpTypeSequenceEmpty},
    {"kOpTypeSequenceErase", kOpTypeSequenceErase},
    {"kOpTypeSequenceInsert", kOpTypeSequenceInsert},
    {"kOpTypeSequenceLength", kOpTypeSequenceLength},
    {"kOpTypeShape", kOpTypeShape},
    {"kOpTypeShrink", kOpTypeShrink},
    {"kOpTypeSigmoid", kOpTypeSigmoid},
    {"kOpTypeSign", kOpTypeSign},
    {"kOpTypeSin", kOpTypeSin},
    {"kOpTypeSinh", kOpTypeSinh},
    {"kOpTypeSize", kOpTypeSize},
    {"kOpTypeSlice", kOpTypeSlice},
    {"kOpTypeSoftmax", kOpTypeSoftmax},
    {"kOpTypeSoftplus", kOpTypeSoftplus},
    {"kOpTypeSoftsign", kOpTypeSoftsign},
    {"kOpTypeSpaceToDepth", kOpTypeSpaceToDepth},
    {"kOpTypeSplit", kOpTypeSplit},
    {"kOpTypeSqrt", kOpTypeSqrt},
    {"kOpTypeSqueeze", kOpTypeSqueeze},
    {"kOpTypeSub", kOpTypeSub},
    {"kOpTypeSum", kOpTypeSum},
    {"kOpTypeTan", kOpTypeTan},
    {"kOpTypeTanh", kOpTypeTanh},
    {"kOpTypeTfIdf", kOpTypeTfIdf},
    {"kOpTypeThresholdedRelu", kOpTypeThresholdedRelu},
    {"kOpTypeTile", kOpTypeTile},
    {"kOpTypeTopK", kOpTypeTopK},
    {"kOpTypeTranspose", kOpTypeTranspose},
    {"kOpTypeUnsqueeze", kOpTypeUnsqueeze},
    {"kOpTypeUpsample", kOpTypeUpsample},
    {"kOpTypeWhere", kOpTypeWhere},
    {"kOpTypeXor", kOpTypeXor},
    {"kOpTypeRMSNorm", kOpTypeRMSNorm},
    {"kOpTypeEmbedding", kOpTypeEmbedding},

    {"kOpTypeNone", kOpTypeNone},
};

std::string opTypeToString(OpType op_type) {
  auto iter = g_optype_string_map.find(op_type);
  if (iter != g_optype_string_map.end()) {
    return iter->second;
  }
  NNDEPLOY_LOGE("Error: op_type=%d not found in g_optype_string_map.\n",
                (int)op_type);
  return std::string();
}

OpType stringToOpType(const std::string &op_type_name) {
  auto iter = g_string_optype_map.find(op_type_name);
  if (iter != g_string_optype_map.end()) {
    return iter->second;
  }
  NNDEPLOY_LOGE("Error: op_type_name=%s not found in g_string_optype_map.\n",
                op_type_name.c_str());
  return kOpTypeNone;
}

std::map<OpType, std::shared_ptr<OpParamCreator>> &
getGlobalOpParamCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<OpType, std::shared_ptr<OpParamCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<OpType, std::shared_ptr<OpParamCreator>>);
  });
  return *creators;
}

std::shared_ptr<base::Param> createOpParam(OpType type) {
  std::shared_ptr<base::Param> temp;
  auto &creater_map = getGlobalOpParamCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createOpParam(type);
  }
  return temp;
}

// Concat 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeConcat, ConcatParam);

// Conv 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeConv, ConvParam);

// MaxPool 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeMaxPool, MaxPoolParam);

// Reshape 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeReshape, ReshapeParam);

// Resize 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeResize, ResizeParam);

// Softmax 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeSoftmax, SoftmaxParam);

// Split 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeSplit, SplitParam);

// Tranpose 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeTranspose, TransposeParam);

// TODO: @Leonisux:
// 补充llama的算子的参数的注册
// RMSNorm 算子参数类的注册函数
REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeRMSNorm, RMSNormParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeBatchNormalization,
                               BatchNormalizationParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeFlatten, FlattenParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeGemm, GemmParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeEmbedding, EmbeddingParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeQuantizeLinear, QuantizeLinearParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeDequantizeLinear, DequantizeLinearParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeQLinearConv, QLinearConvParam);

REGISTER_OP_PARAM_IMPLEMENTION(kOpTypeAveragePool, AvaragePoolParam);

}  // namespace ir
}  // namespace nndeploy