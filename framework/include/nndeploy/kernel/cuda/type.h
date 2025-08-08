#ifndef _NNDEPLOY_KERNEL_CUDA_TYPE_H_
#define _NNDEPLOY_KERNEL_CUDA_TYPE_H_

#define NNDEPLOY_CUDA_PRIMITIVE_REAL_TYPE_SEQ                      \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(float, base::dataTypeOf<float>())     \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(double, base::dataTypeOf<double>())   \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(int32_t, base::dataTypeOf<int32_t>()) \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(int64_t, base::dataTypeOf<int64_t>())

#define NNDEPLOY_CUDA_PRIMITIVE_FLOATING_TYPE_SEQ              \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(float, base::dataTypeOf<float>()) \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(double, base::dataTypeOf<double>())

#define NNDEPLOY_CUDA_PRIMITIVE_INT_TYPE_SEQ                       \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(int32_t, base::dataTypeOf<int32_t>()) \
  NNDEPLOY_PP_MAKE_TUPLE_SEQ(int64_t, base::dataTypeOf<int64_t>())

#endif