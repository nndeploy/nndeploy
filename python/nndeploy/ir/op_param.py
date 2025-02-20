
import nndeploy._nndeploy_internal as _C


name_to_op_type = {
    "Net": _C.ir.OpType.Net,
    "Abs": _C.ir.OpType.Abs,
    "Add": _C.ir.OpType.Add,
    "Acos": _C.ir.OpType.Acos,
    "Adam": _C.ir.OpType.Adam,
    "And": _C.ir.OpType.And,
    "ArgMax": _C.ir.OpType.ArgMax,
    "ArgMin": _C.ir.OpType.ArgMin,
    "Asin": _C.ir.OpType.Asin,
    "Atan": _C.ir.OpType.Atan,
    "AveragePool": _C.ir.OpType.AveragePool,
    "BatchNormalization": _C.ir.OpType.BatchNormalization,
    "Cast": _C.ir.OpType.Cast,
    "Ceil": _C.ir.OpType.Ceil,
    "Clip": _C.ir.OpType.Clip,
    "Concat": _C.ir.OpType.Concat,
    "Constant": _C.ir.OpType.Constant,
    "Conv": _C.ir.OpType.Conv,
    "Cos": _C.ir.OpType.Cos,
    "Cosh": _C.ir.OpType.Cosh,
    "DepthToSpace": _C.ir.OpType.DepthToSpace,
    "DequantizeLinear": _C.ir.OpType.DequantizeLinear,
    "Det": _C.ir.OpType.Det,
    "Div": _C.ir.OpType.Div,
    "Dropout": _C.ir.OpType.Dropout,
    "Einsum": _C.ir.OpType.Einsum,
    "Elu": _C.ir.OpType.Elu,
    "Equal": _C.ir.OpType.Equal,
    "Erf": _C.ir.OpType.Erf,
    "Exp": _C.ir.OpType.Exp,
    "Flatten": _C.ir.OpType.Flatten,
    "Floor": _C.ir.OpType.Floor,
    "Gather": _C.ir.OpType.Gather,
    "Gemm": _C.ir.OpType.Gemm,
    "GlobalAveragePool": _C.ir.OpType.GlobalAveragePool,
    "GlobalLpPool": _C.ir.OpType.GlobalLpPool,
    "GlobalMaxPool": _C.ir.OpType.GlobalMaxPool,
    "Greater": _C.ir.OpType.Greater,
    "HardSigmoid": _C.ir.OpType.HardSigmoid,
    "Identity": _C.ir.OpType.Identity,
    "If": _C.ir.OpType.If,
    "ImageScaler": _C.ir.OpType.ImageScaler,
    "InstanceNormalization": _C.ir.OpType.InstanceNormalization,
    "IsInf": _C.ir.OpType.IsInf,
    "IsNaN": _C.ir.OpType.IsNaN,
    "LRN": _C.ir.OpType.LRN,
    "LSTM": _C.ir.OpType.LSTM,
    "LeakyRelu": _C.ir.OpType.LeakyRelu,
    "Less": _C.ir.OpType.Less,
    "Log": _C.ir.OpType.Log,
    "LogSoftmax": _C.ir.OpType.LogSoftmax,
    "Loop": _C.ir.OpType.Loop,
    "LpNormalization": _C.ir.OpType.LpNormalization,
    "LpPool": _C.ir.OpType.LpPool,
    "MatMul": _C.ir.OpType.MatMul,
    "MatMulInteger": _C.ir.OpType.MatMulInteger,
    "Max": _C.ir.OpType.Max,
    "MaxPool": _C.ir.OpType.MaxPool,
    "MaxRoiPool": _C.ir.OpType.MaxRoiPool,
    "MaxUnpool": _C.ir.OpType.MaxUnpool,
    "Mean": _C.ir.OpType.Mean,
    "Min": _C.ir.OpType.Min,
    "Mod": _C.ir.OpType.Mod,
    "Momentum": _C.ir.OpType.Momentum,
    "Mul": _C.ir.OpType.Mul,
    "Multinomial": _C.ir.OpType.Multinomial,
    "Neg": _C.ir.OpType.Neg,
    "NegLogSoftmax": _C.ir.OpType.NegLogSoftmax,
    "NonMaxSuppression": _C.ir.OpType.NonMaxSuppression,
    "NonZero": _C.ir.OpType.NonZero,
    "Not": _C.ir.OpType.Not,
    "OneHot": _C.ir.OpType.OneHot,
    "OnesLike": _C.ir.OpType.OnesLike,
    "Or": _C.ir.OpType.Or,
    "Pad": _C.ir.OpType.Pad,
    "Pow": _C.ir.OpType.Pow,
    "PRelu": _C.ir.OpType.PRelu,
    "QLinearConv": _C.ir.OpType.QLinearConv,
    "QLinearMatMul": _C.ir.OpType.QLinearMatMul,
    "QuantizeLinear": _C.ir.OpType.QuantizeLinear,
    "RNN": _C.ir.OpType.RNN,
    "RandomNormal": _C.ir.OpType.RandomNormal,
    "RandomNormalLike": _C.ir.OpType.RandomNormalLike,
    "RandomUniform": _C.ir.OpType.RandomUniform,
    "RandomUniformLike": _C.ir.OpType.RandomUniformLike,
    "Range": _C.ir.OpType.Range,
    "Reciprocal": _C.ir.OpType.Reciprocal,
    "ReduceL1": _C.ir.OpType.ReduceL1,
    "ReduceL2": _C.ir.OpType.ReduceL2,
    "ReduceLogSum": _C.ir.OpType.ReduceLogSum,
    "ReduceLogSumExp": _C.ir.OpType.ReduceLogSumExp,
    "ReduceMax": _C.ir.OpType.ReduceMax,
    "ReduceMean": _C.ir.OpType.ReduceMean,
    "ReduceMin": _C.ir.OpType.ReduceMin,
    "ReduceProd": _C.ir.OpType.ReduceProd,
    "ReduceSum": _C.ir.OpType.ReduceSum,
    "ReduceSumSquare": _C.ir.OpType.ReduceSumSquare,
    "Relu": _C.ir.OpType.Relu,
    "Reshape": _C.ir.OpType.Reshape,
    "Resize": _C.ir.OpType.Resize,
    "ReverseSequence": _C.ir.OpType.ReverseSequence,
    "RoiAlign": _C.ir.OpType.RoiAlign,
    "Round": _C.ir.OpType.Round,
    "Scale": _C.ir.OpType.Scale,
    "Scan": _C.ir.OpType.Scan,
    "Scatter": _C.ir.OpType.Scatter,
    "Selu": _C.ir.OpType.Selu,
    "SequenceAt": _C.ir.OpType.SequenceAt,
    "SequenceConstruct": _C.ir.OpType.SequenceConstruct,
    "SequenceEmpty": _C.ir.OpType.SequenceEmpty,
    "SequenceErase": _C.ir.OpType.SequenceErase,
    "SequenceInsert": _C.ir.OpType.SequenceInsert,
    "SequenceLength": _C.ir.OpType.SequenceLength,
    "Shape": _C.ir.OpType.Shape,
    "Shrink": _C.ir.OpType.Shrink,
    "Sigmoid": _C.ir.OpType.Sigmoid,
    "Sign": _C.ir.OpType.Sign,
    "Sin": _C.ir.OpType.Sin,
    "Sinh": _C.ir.OpType.Sinh,
    "Size": _C.ir.OpType.Size,
    "Slice": _C.ir.OpType.Slice,
    "Softmax": _C.ir.OpType.Softmax,
    "Softplus": _C.ir.OpType.Softplus,
    "Softsign": _C.ir.OpType.Softsign,
    "SpaceToDepth": _C.ir.OpType.SpaceToDepth,
    "Split": _C.ir.OpType.Split,
    "Sqrt": _C.ir.OpType.Sqrt,
    "Squeeze": _C.ir.OpType.Squeeze,
    "Sub": _C.ir.OpType.Sub,
    "Sum": _C.ir.OpType.Sum,
    "Tan": _C.ir.OpType.Tan,
    "Tanh": _C.ir.OpType.Tanh,
    "TfIdf": _C.ir.OpType.TfIdf,
    "ThresholdedRelu": _C.ir.OpType.ThresholdedRelu,
    "Tile": _C.ir.OpType.Tile,
    "TopK": _C.ir.OpType.TopK,
    "Transpose": _C.ir.OpType.Transpose,
    "Unsqueeze": _C.ir.OpType.Unsqueeze,
    "Upsample": _C.ir.OpType.Upsample,
    "Where": _C.ir.OpType.Where,
    "Xor": _C.ir.OpType.Xor,
    "RMSNorm": _C.ir.OpType.RMSNorm,
    "Embedding": _C.ir.OpType.Embedding,
    "None": _C.ir.OpType.kOpTypeNone,
}


op_type_to_name = {v: k for k, v in name_to_op_type.items()}


def op_type_to_string(op_type):
    return op_type_to_name[op_type]


def string_to_op_type(op_type_name):
    return name_to_op_type[op_type_name]
    

class OpType(_C.ir.OpType):
    Net = _C.ir.OpType.Net
    Abs = _C.ir.OpType.Abs
    Add = _C.ir.OpType.Add
    Acos = _C.ir.OpType.Acos
    Adam = _C.ir.OpType.Adam
    And = _C.ir.OpType.And
    ArgMax = _C.ir.OpType.ArgMax
    ArgMin = _C.ir.OpType.ArgMin
    Asin = _C.ir.OpType.Asin
    Atan = _C.ir.OpType.Atan
    AveragePool = _C.ir.OpType.AveragePool
    BatchNormalization = _C.ir.OpType.BatchNormalization
    Cast = _C.ir.OpType.Cast
    Ceil = _C.ir.OpType.Ceil
    Clip = _C.ir.OpType.Clip
    Concat = _C.ir.OpType.Concat
    Constant = _C.ir.OpType.Constant
    Conv = _C.ir.OpType.Conv
    Cos = _C.ir.OpType.Cos
    Cosh = _C.ir.OpType.Cosh
    DepthToSpace = _C.ir.OpType.DepthToSpace
    DequantizeLinear = _C.ir.OpType.DequantizeLinear
    Det = _C.ir.OpType.Det
    Div = _C.ir.OpType.Div
    Dropout = _C.ir.OpType.Dropout
    Einsum = _C.ir.OpType.Einsum
    Elu = _C.ir.OpType.Elu
    Equal = _C.ir.OpType.Equal
    Erf = _C.ir.OpType.Erf
    Exp = _C.ir.OpType.Exp
    Flatten = _C.ir.OpType.Flatten
    Floor = _C.ir.OpType.Floor
    Gather = _C.ir.OpType.Gather
    Gemm = _C.ir.OpType.Gemm
    GlobalAveragePool = _C.ir.OpType.GlobalAveragePool
    GlobalLpPool = _C.ir.OpType.GlobalLpPool
    GlobalMaxPool = _C.ir.OpType.GlobalMaxPool
    Greater = _C.ir.OpType.Greater
    HardSigmoid = _C.ir.OpType.HardSigmoid
    Identity = _C.ir.OpType.Identity
    If = _C.ir.OpType.If
    ImageScaler = _C.ir.OpType.ImageScaler
    InstanceNormalization = _C.ir.OpType.InstanceNormalization
    IsInf = _C.ir.OpType.IsInf
    IsNaN = _C.ir.OpType.IsNaN
    LRN = _C.ir.OpType.LRN
    LSTM = _C.ir.OpType.LSTM
    LeakyRelu = _C.ir.OpType.LeakyRelu
    Less = _C.ir.OpType.Less
    Log = _C.ir.OpType.Log
    LogSoftmax = _C.ir.OpType.LogSoftmax
    Loop = _C.ir.OpType.Loop
    LpNormalization = _C.ir.OpType.LpNormalization
    LpPool = _C.ir.OpType.LpPool
    MatMul = _C.ir.OpType.MatMul
    MatMulInteger = _C.ir.OpType.MatMulInteger
    Max = _C.ir.OpType.Max
    MaxPool = _C.ir.OpType.MaxPool
    MaxRoiPool = _C.ir.OpType.MaxRoiPool
    MaxUnpool = _C.ir.OpType.MaxUnpool
    Mean = _C.ir.OpType.Mean
    Min = _C.ir.OpType.Min
    Mod = _C.ir.OpType.Mod
    Momentum = _C.ir.OpType.Momentum
    Mul = _C.ir.OpType.Mul
    Multinomial = _C.ir.OpType.Multinomial
    Neg = _C.ir.OpType.Neg
    NegLogSoftmax = _C.ir.OpType.NegLogSoftmax
    NonMaxSuppression = _C.ir.OpType.NonMaxSuppression
    NonZero = _C.ir.OpType.NonZero
    Not = _C.ir.OpType.Not
    OneHot = _C.ir.OpType.OneHot
    OnesLike = _C.ir.OpType.OnesLike
    Or = _C.ir.OpType.Or
    Pad = _C.ir.OpType.Pad
    Pow = _C.ir.OpType.Pow
    PRelu = _C.ir.OpType.PRelu
    QLinearConv = _C.ir.OpType.QLinearConv
    QLinearMatMul = _C.ir.OpType.QLinearMatMul
    QuantizeLinear = _C.ir.OpType.QuantizeLinear
    RNN = _C.ir.OpType.RNN
    RandomNormal = _C.ir.OpType.RandomNormal
    RandomNormalLike = _C.ir.OpType.RandomNormalLike
    RandomUniform = _C.ir.OpType.RandomUniform
    RandomUniformLike = _C.ir.OpType.RandomUniformLike
    Range = _C.ir.OpType.Range
    Reciprocal = _C.ir.OpType.Reciprocal
    ReduceL1 = _C.ir.OpType.ReduceL1
    ReduceL2 = _C.ir.OpType.ReduceL2
    ReduceLogSum = _C.ir.OpType.ReduceLogSum
    ReduceLogSumExp = _C.ir.OpType.ReduceLogSumExp
    ReduceMax = _C.ir.OpType.ReduceMax
    ReduceMean = _C.ir.OpType.ReduceMean
    ReduceMin = _C.ir.OpType.ReduceMin
    ReduceProd = _C.ir.OpType.ReduceProd
    ReduceSum = _C.ir.OpType.ReduceSum
    ReduceSumSquare = _C.ir.OpType.ReduceSumSquare
    Relu = _C.ir.OpType.Relu
    Reshape = _C.ir.OpType.Reshape
    Resize = _C.ir.OpType.Resize
    ReverseSequence = _C.ir.OpType.ReverseSequence
    RoiAlign = _C.ir.OpType.RoiAlign
    Round = _C.ir.OpType.Round
    Scale = _C.ir.OpType.Scale
    Scan = _C.ir.OpType.Scan
    Scatter = _C.ir.OpType.Scatter
    Selu = _C.ir.OpType.Selu
    SequenceAt = _C.ir.OpType.SequenceAt
    SequenceConstruct = _C.ir.OpType.SequenceConstruct
    SequenceEmpty = _C.ir.OpType.SequenceEmpty
    SequenceErase = _C.ir.OpType.SequenceErase
    SequenceInsert = _C.ir.OpType.SequenceInsert
    SequenceLength = _C.ir.OpType.SequenceLength
    Shape = _C.ir.OpType.Shape
    Shrink = _C.ir.OpType.Shrink
    Sigmoid = _C.ir.OpType.Sigmoid
    Sign = _C.ir.OpType.Sign
    Sin = _C.ir.OpType.Sin
    Sinh = _C.ir.OpType.Sinh
    Size = _C.ir.OpType.Size
    Slice = _C.ir.OpType.Slice
    Softmax = _C.ir.OpType.Softmax
    Softplus = _C.ir.OpType.Softplus
    Softsign = _C.ir.OpType.Softsign
    SpaceToDepth = _C.ir.OpType.SpaceToDepth
    Split = _C.ir.OpType.Split
    Sqrt = _C.ir.OpType.Sqrt
    Squeeze = _C.ir.OpType.Squeeze
    Sub = _C.ir.OpType.Sub
    Sum = _C.ir.OpType.Sum
    Tan = _C.ir.OpType.Tan
    Tanh = _C.ir.OpType.Tanh
    TfIdf = _C.ir.OpType.TfIdf
    ThresholdedRelu = _C.ir.OpType.ThresholdedRelu
    Tile = _C.ir.OpType.Tile
    TopK = _C.ir.OpType.TopK
    Transpose = _C.ir.OpType.Transpose
    Unsqueeze = _C.ir.OpType.Unsqueeze
    Upsample = _C.ir.OpType.Upsample
    Where = _C.ir.OpType.Where
    Xor = _C.ir.OpType.Xor
    RMSNorm = _C.ir.OpType.RMSNorm
    Embedding = _C.ir.OpType.Embedding
    kOpTypeNone = _C.ir.OpType.kOpTypeNone
    
    @classmethod
    def from_name(cls, op_type_name: str):
        if op_type_name not in name_to_op_type:
            raise ValueError(f"not supported op type name: {op_type_name}")
        return cls(name_to_op_type[op_type_name])


class OpParamCreator(_C.ir.OpParamCreator):
    def __init__(self):
        super().__init__()

    def create_op_param(self, op_type: OpType):
        raise NotImplementedError("base class OpParamCreator does not implement create_op_param method")


def register_op_param_creator(op_type: OpType, creator: OpParamCreator):
    _C.ir.register_op_param_creator(op_type, creator)


def create_op_param(op_type: OpType):
    return _C.ir.create_op_param(op_type)


class OpParam(_C.ir.OpParam):
    def __init__(self):
        super().__init__()