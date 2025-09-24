#ifndef _NNDEPLOY_KERNEL_ELEMENTWISE_UNARY_KERNEL_H_
#define _NNDEPLOY_KERNEL_ELEMENTWISE_UNARY_KERNEL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/device.h"
#include "nndeploy/kernel/kernel.h"
#include "nndeploy/kernel/scalar.h"

namespace nndeploy {
namespace kernel {

#define NNDEPLOY_UNARY_FLOATING_MATH_KERNEL_SEQ                           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeRelu)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeElu)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeCelu)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeHardSwish)       \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeHardSigmoid)     \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeHardShrink)      \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeHardTanh)        \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLeakyRelu)       \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeMish)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSelu)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSilu)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSoftShrink)      \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSoftSign)        \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSoftPlus)        \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeTanh)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeThreshold)       \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAbs)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAcos)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAcosh)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAsin)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAsinh)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAtan)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeAtanh)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeCeil)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeCos)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeCosh)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeErf)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeErfc)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeExp)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeExp2)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeExpm1)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeFloor)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLgamma)          \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLog)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLog2)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLog10)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLog1p)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLogSigmoid)      \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeNegative)        \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeReciprocal)      \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeReciprocalNoNan) \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeRint)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeRound)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeRsqrt)           \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSigmoid)         \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSign)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSin)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSinh)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSqrt)            \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSquare)          \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeTan)             \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeNotEqualZero)    \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeFastGelu)        \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeQuickGelu)       \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeSquareReLU)

#define NNDEPLOY_UNARY_LOGICAL_KERNEL_SEQ \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeLogicalNot)

#define NNDEPLOY_UNARY_BITWISE_KERNEL_SEQ \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(UnaryKernelType::kKernelTypeBitwiseNot)

class ElementwiseUnaryKernel : public Kernel {
 public:
  ElementwiseUnaryKernel() = default;
  ~ElementwiseUnaryKernel() override = default;

  virtual void Launch(device::Stream* stream, const void* src, void* dst,
                      size_t count) = 0;
};

class ElementwiseUnaryKernelFactory : public Factory<ElementwiseUnaryKernel> {
 public:
  ElementwiseUnaryKernelFactory() = default;
  ~ElementwiseUnaryKernelFactory() override = default;

  virtual std::unique_ptr<ElementwiseUnaryKernel> New(
      UnaryKernelType kernel, base::DataType src_type,
      base::DataType dst_type) = 0;

  virtual std::unique_ptr<ElementwiseUnaryKernel> New(UnaryKernelType op,
                                                      base::DataType src_type,
                                                      base::DataType dst_type,
                                                      Scalar attr0) = 0;

  virtual std::unique_ptr<ElementwiseUnaryKernel> New(UnaryKernelType op,
                                                      base::DataType src_type,
                                                      base::DataType dst_type,
                                                      Scalar attr0,
                                                      Scalar attr1) = 0;
};

}  // namespace kernel
}  // namespace nndeploy

#endif /*  */
