#include "nndeploy/device/cuda/cuda_device.h"
#include "nndeploy/kernel/elementwise_unary_kernel.h"
#include "nndeploy/kernel/util.h"
#include "nndeploy/kernel/cuda/type.h"
namespace nndeploy {
namespace kernel {
namespace {
template <UnaryKernelType unary_kernel, typename Src, typename Dst>
class CudaElementwiseUnaryImpl : public ElementwiseUnaryKernel {
 public:
  CudaElementwiseUnaryImpl(Scalar attr0, Scalar attr1)
      : attr0(attr0), attr1(attr1) {}
  ~CudaElementwiseUnaryImpl() override = default;

  void Launch(device::Stream* stream, const void* src, void* dst,
              size_t count) override {
    // cudaStream_t cuda_stream =
    //     (cudaStream_t)stream->as<CudaStream>()->getStream();
    // auto functor =
    //     UnaryFunctor<DeviceType::kCUDA, unary_kernel, Dst, Src>(attr0,
    //     attr1);

    // TODO: sjx  cuda执行函数
    // OF_CUDA_CHECK((cuda::elementwise::Unary<decltype(functor), Dst, Src>(
    //     functor, count, reinterpret_cast<Dst*>(dst), reinterpret_cast<const
    //     Src*>(src), cuda_stream->cuda_stream())));
  }

 protected:
  Scalar attr0, attr1;
};

template <UnaryKernelType unary_kernel, typename Src, typename Dst>
std::unique_ptr<ElementwiseUnaryKernel> NewCudaElementwiseUnaryKernel(
    Scalar attr0, Scalar attr1) {
  return std::unique_ptr<ElementwiseUnaryKernel>(
      new CudaElementwiseUnaryImpl<unary_kernel, Src, Dst>(attr0, attr1));
};

class CudaElementwiseUnaryFactory : public ElementwiseUnaryKernelFactory {
 public:
  CudaElementwiseUnaryFactory() = default;

  std::unique_ptr<ElementwiseUnaryKernel> New(
      UnaryKernelType kernel, base::DataType src_type,
      base::DataType dst_type) override {
    return New(kernel, src_type, dst_type, Scalar(), Scalar());
  }

  std::unique_ptr<ElementwiseUnaryKernel> New(UnaryKernelType kernel,
                                              base::DataType src_type,
                                              base::DataType dst_type,
                                              Scalar attr0) override {
    return New(kernel, src_type, dst_type, attr0, Scalar());
  }

  std::unique_ptr<ElementwiseUnaryKernel> New(UnaryKernelType kernel,
                                              base::DataType src_type,
                                              base::DataType dst_type,
                                              Scalar attr0,
                                              Scalar attr1) override {
    // Register all supported kernelerations
#define MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY(kernel, dtype_pair)      \
  {std::make_tuple(kernel, NNDEPLOY_PP_PAIR_SECOND(dtype_pair),              \
                   NNDEPLOY_PP_PAIR_SECOND(dtype_pair)),                     \
   NewCudaElementwiseUnaryKernel<kernel, NNDEPLOY_PP_PAIR_FIRST(dtype_pair), \
                                 NNDEPLOY_PP_PAIR_FIRST(dtype_pair)>},

#define MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY(                 \
    kernel, src_type_pair, dst_dtype_pair)                                \
  \ {std::make_tuple(kernel, NNDEPLOY_PP_PAIR_SECOND(src_type_pair),      \
                     NNDEPLOY_PP_PAIR_SECOND(dst_dtype_pair)),            \
     NewCudaElementwiseUnaryKernel<kernel,                                \
                                   NNDEPLOY_PP_PAIR_FIRST(src_type_pair), \
                                   NNDEPLOY_PP_PAIR_FIRST(dst_dtype_pair)>},

    static const std::map<
        std::tuple<UnaryKernelType, base::DataType, base::DataType>,
        std::function<std::unique_ptr<ElementwiseUnaryKernel>(Scalar, Scalar)>>
        new_elementwise_unary_handle{

            // For Float Type kernel
            NNDEPLOY_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                NNDEPLOY_UNARY_FLOATING_MATH_KERNEL_SEQ,
                NNDEPLOY_CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)

        };

#undef MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY
#undef MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY

    const auto it = new_elementwise_unary_handle.find(
        std::make_tuple(kernel, src_type, dst_type));
    if (it != new_elementwise_unary_handle.end()) {
      return it->second(attr0, attr1);
    } else {
      // TODO:sjx  加一句报错
      return nullptr;
    }
  }
};

REGISTER_KERNEL_FACTORY(base::kDeviceTypeCodeCuda,
                        ElementwiseUnaryKernelFactory,
                        CudaElementwiseUnaryFactory);
}  // namespace
}  // namespace kernel
}  // namespace nndeploy
