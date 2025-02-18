
#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/half.hpp"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_cast.h"

namespace nndeploy {
namespace op {

namespace cast_type {

template <typename I, typename O>
void impl(void *input, void *output) {
  *(O *)output = static_cast<O>((*(I *)input));
}

void impl_float_2_bfloat16(void *input, void *output) {
  bfp16_t *bfp16ptr = (bfp16_t *)output;
  float *floatptr = (float *)input;
  *bfp16ptr = *floatptr;
}

void impl_bf16_2_float(void *input, void *output) {
  bfp16_t *bfp16ptr = (bfp16_t *)input;
  float *floatptr = (float *)output;
  *floatptr = *bfp16ptr;
}

static const float MAX_HALF_FLOAT = 65504.0f;
static const float MIN_HALF_FLOAT = -65504.0f;
void impl_half_2_float(void *input, void *output) {
  half_float::detail::uint16 *fp16ptr = (half_float::detail::uint16 *)input;
  float *float_ptr = (float *)output;
  *float_ptr = half_float::detail::half2float<float>(*fp16ptr);
}

void impl_float_2_half(void *input, void *output) {
  half_float::detail::uint16 *fp16ptr = (half_float::detail::uint16 *)input;
  float *float_ptr = (float *)input;
  if (*float_ptr > MAX_HALF_FLOAT) {
    NNDEPLOY_LOGE(
        "ERROR: the weights=%f is out of bounds "
        "of float16 max %f. \n",
        *float_ptr, MAX_HALF_FLOAT);
    *fp16ptr = half_float::detail::float2half<(std::float_round_style)(
        HALF_ROUND_STYLE)>(MAX_HALF_FLOAT);
  } else if (*float_ptr < MIN_HALF_FLOAT) {
    NNDEPLOY_LOGE(
        "ERROR: the weights=%f is out of bounds "
        "of float16 min %f. \n",
        *float_ptr, MIN_HALF_FLOAT);
    *fp16ptr = half_float::detail::float2half<(std::float_round_style)(
        HALF_ROUND_STYLE)>(MIN_HALF_FLOAT);
  } else {
    *fp16ptr = half_float::detail::float2half<(std::float_round_style)(
        HALF_ROUND_STYLE)>(*float_ptr);
  }
}

struct CastHash {
  CastHash(base::DataType from, base::DataType to) : from(from), to(to) {}
  base::DataType from;
  base::DataType to;
  uint32_t hash_(base::DataType dtype) {
    return dtype.code_ << 24 | dtype.bits_ << 16 | dtype.lanes_;
  }
  uint64_t hash() { return uint64_t(hash_(from)) << 32 | hash_(to); }
};

#define MAP_INSERT_SELF(S)                                                   \
  map.insert({CastHash(base::dataTypeOf<S>(), base::dataTypeOf<S>()).hash(), \
              impl<S, S>});

#define MAP_INSERT(I, O)                                                     \
  map.insert({CastHash(base::dataTypeOf<I>(), base::dataTypeOf<O>()).hash(), \
              impl<I, O>});                                                  \
  map.insert({CastHash(base::dataTypeOf<O>(), base::dataTypeOf<I>()).hash(), \
              impl<O, I>});

#define MAP_INSERT_FUNC(I, O, FUNC) \
  map.insert(                       \
      {CastHash(base::dataTypeOf<I>(), base::dataTypeOf<O>()).hash(), FUNC});

void (*getCastImpl(base::DataType from, base::DataType to))(void *, void *) {
  static std::unordered_map<uint64_t, void (*)(void *, void *)> map;
  auto init = [&]() {
    MAP_INSERT_SELF(bool);
    MAP_INSERT_SELF(int8_t);
    MAP_INSERT_SELF(int16_t);
    MAP_INSERT_SELF(int32_t);
    MAP_INSERT_SELF(int64_t);
    MAP_INSERT_SELF(float);
    MAP_INSERT_SELF(double);
    MAP_INSERT_SELF(bfp16_t);
    MAP_INSERT_SELF(half_float::half);

    MAP_INSERT(bool, int8_t);
    MAP_INSERT(int8_t, uint8_t);
    MAP_INSERT(int16_t, uint16_t);
    MAP_INSERT(int32_t, uint32_t);
    MAP_INSERT(bool, float);
    MAP_INSERT(int32_t, float);
    MAP_INSERT(int32_t, double);
    MAP_INSERT(int64_t, float);
    MAP_INSERT(int64_t, double);
    MAP_INSERT_FUNC(float, bfp16_t, impl_float_2_bfloat16);
    MAP_INSERT_FUNC(bfp16_t, float, impl_float_2_bfloat16);
    MAP_INSERT_FUNC(float, half_float::half, impl_float_2_half);
    MAP_INSERT_FUNC(half_float::half, float, impl_half_2_float);
  };

  std::once_flag on;
  std::call_once(on, init);

  auto it = map.find(CastHash(from, to).hash());
  if (it == map.end()) {
    NNDEPLOY_LOGE("cast not support from %s to %s",
                  base::dataTypeToString(from).c_str(),
                  base::dataTypeToString(to).c_str());
  }

  return it->second;
}

};  // namespace cast_type

base::Status OpCast::inferShape() {
  auto input_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(input_shape);
  auto param = dynamic_cast<ir::CastParam *>(op_desc_.op_param_.get());
  outputs_[0]->setDataType(param->cast_type);
  return base::kStatusCodeOk;
}

base::Status OpCast::run() {
  device::Tensor *input_tensor = inputs_[0];
  device::Tensor *output_tensor = outputs_[0];
  void *input_ptr = input_tensor->getData();
  void *output_ptr = output_tensor->getData();

  auto param = dynamic_cast<ir::CastParam *>(op_desc_.op_param_.get());
  auto shape = input_tensor->getShape();
  int total_elements =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  base::DataType dtype = param->cast_type;
  base::DataType input_dtype = input_tensor->getDataType();
  base::DataType out_dtype = output_tensor->getDataType();
  if (out_dtype != dtype) {
    output_tensor->setDataType(dtype);
  }

  auto impl = cast_type::getCastImpl(input_dtype, dtype);

  for (int i = 0; i < total_elements; i++) {
    impl(input_ptr, output_ptr);
  }
  return base::kStatusCodeOk;
}

base::Status cast(device::Tensor *input, device::Tensor *output,
                  std::shared_ptr<base::Param> param) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input->getDeviceType(), "", ir::kOpTypeCast);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  delete op;

  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeCast, OpCast)

}  // namespace op
}  // namespace nndeploy
