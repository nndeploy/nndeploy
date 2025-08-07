#ifndef _NNDEPLOY_KERNEL_UNARY_FUNCTOR_H_
#define _NNDEPLOY_KERNEL_UNARY_FUNCTOR_H_

#include <cmath>

#include "nndeploy/kernel/kernel.h"
#include "nndeploy/kernel/scalar.h"
#include "nndeploy/kernel/util.h"

namespace nndeploy {
namespace kernel {

template <int Device, UnaryKernelType kernel, typename Dst, typename Src>
struct UnaryFunctor;

// Relu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeRelu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src zero = Src(0);
    return static_cast<Dst>(src > zero ? src : zero);
  }
};

// Elu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeElu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar)
      : alpha(a0.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src > Src(0) ? src
                                         : alpha * (std::exp(src) - Src(1)));
  }
  const double alpha;
};

// Celu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeCelu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar)
      : alpha(a0.Value<double>()), inv_alpha(1.0 / a0.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(
        src > Src(0) ? src : alpha * (std::exp(src * inv_alpha) - Src(1)));
  }
  const double alpha;
  const double inv_alpha;
};

// HardSwish
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeHardSwish, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src three = Src(3);
    const Src six = Src(6);
    return static_cast<Dst>(src <= -three  ? Src(0)
                            : src >= three ? src
                                           : src * (src + three) / six);
  }
};

// HardSigmoid
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeHardSigmoid, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src three = Src(3);
    const Src six = Src(6);
    const Src half = Src(0.5);
    return static_cast<Dst>(src <= -three  ? Src(0)
                            : src >= three ? Src(1)
                                           : src / six + half);
  }
};

// HardShrink
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeHardShrink, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar)
      : lambd(a0.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return (std::abs(src) <= lambd) ? Dst(0) : static_cast<Dst>(src);
  }
  const double lambd;
};

// HardTanh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeHardTanh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar a1)
      : min_val(a0.Value<double>()), max_val(a1.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src <= min_val   ? min_val
                            : src >= max_val ? max_val
                                             : src);
  }
  const double min_val;
  const double max_val;
};

// LeakyRelu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLeakyRelu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar)
      : alpha(a0.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src > Src(0) ? src : alpha * src);
  }
  const double alpha;
};

// Mish
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeMish, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    using std::exp;
    using std::tanh;
    Src soft_plus = std::log(Src(1) + exp(src));
    return static_cast<Dst>(src * tanh(soft_plus));
  }
};

// Selu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSelu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const double scale = 1.0507009873554804934193349852946;
    const double alpha = 1.6732632423543772848170429916717;
    return static_cast<Dst>(
        src > Src(0) ? scale * src : scale * alpha * (std::exp(src) - Src(1)));
  }
};

// Silu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSilu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src / (Src(1) + std::exp(-src)));
  }
};

// SoftSign
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSoftSign, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src / (Src(1) + std::abs(src)));
  }
};

// SoftPlus
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSoftPlus, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar a1)
      : beta(a0.Value<double>()), threshold(a1.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>((src * beta) > threshold
                                ? src
                                : std::log(Src(1) + std::exp(src * beta)) /
                                      beta);
  }
  const double beta;
  const double threshold;
};

// SoftShrink
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSoftShrink, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar)
      : alpha(a0.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::abs(src) <= alpha ? Src(0)
                            : src > alpha          ? src - alpha
                                                   : src + alpha);
  }
  const double alpha;
};

// Tanh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeTanh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::tanh(src));
  }
};

// Threshold
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeThreshold, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar a0, Scalar a1)
      : threshold(a0.Value<double>()), value(a1.Value<double>()) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src <= threshold ? value : src);
  }
  const double threshold;
  const double value;
};

// FastGelu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeFastGelu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src a = Src(0.7978845608028654);
    const Src b = Src(0.044715);
    const Src c = Src(0.5);
    const Src x = a * (src + b * src * src * src);
    return static_cast<Dst>(src * (Src(0.5) * (Src(1) + std::tanh(x))));
  }
};

// QuickGelu
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeQuickGelu, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src pi_over_sqrt_8 = Src(1.1107207345395915617539702475152);
    return static_cast<Dst>(
        src * (Src(0.5) * (Src(1) + std::tanh(pi_over_sqrt_8 * src))));
  }
};

// SquareReLU
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSquareReLU, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src zero = Src(0);
    return static_cast<Dst>(src > zero ? src * src : zero);
  }
};

// Abs
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAbs, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::abs(src));
  }
};

// Acos
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAcos, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::acos(src));
  }
};

// Acosh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAcosh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::acosh(src));
  }
};

// Asin
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAsin, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::asin(src));
  }
};

// Asinh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAsinh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::asinh(src));
  }
};

// Atan
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAtan, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::atan(src));
  }
};

// Atanh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeAtanh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::atanh(src));
  }
};

// Ceil
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeCeil, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::ceil(src));
  }
};

// Cos
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeCos, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::cos(src));
  }
};

// Cosh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeCosh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::cosh(src));
  }
};

// Erf
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeErf, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::erf(src));
  }
};

// Erfc
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeErfc, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::erfc(src));
  }
};

// Exp
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeExp, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::exp(src));
  }
};

// Exp2
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeExp2, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::exp2(src));
  }
};

// Expm1
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeExpm1, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::expm1(src));
  }
};

// Floor
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeFloor, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::floor(src));
  }
};

// Lgamma
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLgamma, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::lgamma(src));
  }
};

// Log
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLog, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::log(src));
  }
};

// Log2
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLog2, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::log2(src));
  }
};

// Log10
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLog10, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::log10(src));
  }
};

// Log1p
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLog1p, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::log1p(src));
  }
};

// LogSigmoid
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLogSigmoid, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(-std::log(Src(1.0) + std::exp(-src)));
  }
};

// Negative
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeNegative, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(-src);
  }
};

// Reciprocal
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeReciprocal, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(Src(1.0) / src);
  }
};

// ReciprocalNoNan
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeReciprocalNoNan, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return (std::abs(src) <= Src(0.0)) ? Dst(0.0)
                                       : static_cast<Dst>(Src(1.0) / src);
  }
};

// Rint
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeRint, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::rint(src));
  }
};

// Round
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeRound, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::round(src));
  }
};

// Rsqrt
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeRsqrt, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(Src(1.0) / std::sqrt(src));
  }
};

// Sigmoid
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSigmoid, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(Src(1.0) / (Src(1.0) + std::exp(-src)));
  }
};

// Sign
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSign, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    const Src zero = Src(0);
    return static_cast<Dst>(src > zero ? 1.0 : src < zero ? -1.0 : 0.0);
  }
};

// Sin
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSin, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::sin(src));
  }
};

// Sinh
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSinh, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::sinh(src));
  }
};

// Sqrt
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSqrt, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::sqrt(src));
  }
};

// Square
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeSquare, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src * src);
  }
};

// Tan
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeTan, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(std::tan(src));
  }
};

// NotEqualZero
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeNotEqualZero, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src != Src(0));
  }
};

// LogicalNot
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeLogicalNot, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(!src);
  }
};

// Cast
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeCast, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src);
  }
};

// BitwiseNot
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeBitwiseNot, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(~src);
  }
};

// ------- 复数专用（仅对 std::complex<T> 生效） -------
template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeConj, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return Dst{src.real(), -src.imag()};
  }
};

template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeReal, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src.real());
  }
};

template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeImag, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Dst>(src.imag());
  }
};

template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeRealGrad, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return Dst{src, typename Dst::value_type(0)};
  }
};

template <int Device, typename Dst, typename Src>
struct UnaryFunctor<Device, kKernelTypeImagGrad, Dst, Src> {
  NNDEPLOY_DEVICE_FUNC UnaryFunctor(Scalar, Scalar) {}
  NNDEPLOY_DEVICE_FUNC Dst operator()(Src src) const {
    return Dst{typename Dst::value_type(0), src};
  }
};

// ------- 工具函数：判断是否支持 -------
template <int Device, UnaryKernelType Op, typename Dst, typename Src>
inline constexpr bool IsUnaryFunctorDefined = true;

}  // namespace kernel
}  // namespace nndeploy

#undef NNDEPLOY_DEVICE_FUNC
#endif  // _NNDEPLOY_KERNEL_ELEMENTWISE_UNARY_FUNCTOR_H_