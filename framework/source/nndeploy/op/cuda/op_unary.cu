#include "nndeploy/kernel/elementwise_unary_kernel.h"
#include "nndeploy/kernel/kernel.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

using namespace nndeploy::kernel;
using namespace nndeploy::ir;

template <kernel::UnaryKernelType Kernel>
class CudaOpElementwiseUnary : public OpUnary {
 public:
  CudaOpElementwiseUnary() : OpUnary() {}

  virtual ~CudaOpElementwiseUnary() {}

  virtual base::Status init() {
    OpUnary::init();

    unary_kernel_ = kernel::NewKernel<kernel::ElementwiseUnaryKernelFactory>(
        device_type_.code_, Kernel, inputs_[0]->getDataType(),
        outputs_[0]->getDataType());

    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    auto input_shape = inputs_[0]->getShape();
    int64_t input_elements = std::accumulate(
        input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    unary_kernel_->Launch(stream_, inputs_[0]->getData(),
                          outputs_[0]->getData(), input_elements);

    return base::kStatusCodeOk;
  }

 private:
  std::unique_ptr<kernel::ElementwiseUnaryKernel> unary_kernel_;
};

// 修改宏定义，使用中间类型别名避免模板参数拼接问题
#define REGISTER_CUDA_ELEMENTWISE_OP(op, kernel)           \
  using Cuda##op##kernel = CudaOpElementwiseUnary<kernel>; \
  REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCuda, op, Cuda##op##kernel)

REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeRelu, kKernelTypeRelu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeElu, kKernelTypeElu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeCelu, kKernelTypeCelu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeHardSwish, kKernelTypeHardSwish)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeHardSigmoid, kKernelTypeHardSigmoid)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeHardShrink, kKernelTypeHardShrink)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeHardTanh, kKernelTypeHardTanh)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLeakyRelu, kKernelTypeLeakyRelu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeMish, kKernelTypeMish)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSelu, kKernelTypeSelu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSilu, kKernelTypeSilu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSoftShrink, kKernelTypeSoftShrink)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSoftSign, kKernelTypeSoftSign)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSoftPlus, kKernelTypeSoftPlus)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeTanh, kKernelTypeTanh)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeThreshold, kKernelTypeThreshold)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAbs, kKernelTypeAbs)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAcos, kKernelTypeAcos)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAcosh, kKernelTypeAcosh)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAsin, kKernelTypeAsin)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAsinh, kKernelTypeAsinh)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAtan, kKernelTypeAtan)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeAtanh, kKernelTypeAtanh)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeCeil, kKernelTypeCeil)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeCos, kKernelTypeCos)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeCosh, kKernelTypeCosh)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeErf, kKernelTypeErf)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeErfc, kKernelTypeErfc)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeExp, kKernelTypeExp)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeExp2, kKernelTypeExp2)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeExpm1, kKernelTypeExpm1)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeFloor, kKernelTypeFloor)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLgamma, kKernelTypeLgamma)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLog, kKernelTypeLog)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLog2, kKernelTypeLog2)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLog10, kKernelTypeLog10)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLog1p, kKernelTypeLog1p)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLogSigmoid, kKernelTypeLogSigmoid)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeNegative, kKernelTypeNegative)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeReciprocal, kKernelTypeReciprocal)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeReciprocalNoNan,
// kKernelTypeReciprocalNoNan) REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeRint,
// kKernelTypeRint)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeRound, kKernelTypeRound)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeRsqrt, kKernelTypeRsqrt)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSigmoid, kKernelTypeSigmoid)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSign, kKernelTypeSign)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSin, kKernelTypeSin)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSinh, kKernelTypeSinh)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSqrt, kKernelTypeSqrt)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSquare, kKernelTypeSquare)
REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeTan, kKernelTypeTan)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeNotEqualZero, kKernelTypeNotEqualZero)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeFastGelu, kKernelTypeFastGelu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeQuickGelu, kKernelTypeQuickGelu)
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeSquareReLU, kKernelTypeSquareReLU)

// 逻辑运算算子
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeLogicalNot, kKernelTypeLogicalNot)

// 位运算算子
// REGISTER_CUDA_ELEMENTWISE_OP(kOpTypeBitwiseNot, kKernelTypeBitwiseNot)

}  // namespace op
}  // namespace nndeploy