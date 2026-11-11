
#include "nndeploy/op/op_gather.h"
#include "nndeploy/base/shape.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
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

namespace nndeploy {
namespace op {

base::Status OpGather::run() {
  const auto shape = inputs_[0]->getShape();
  const auto param = static_cast<ir::GatherParam *>(op_desc_.op_param_.get());
  int axis = param ? param->axis_ : 0;
  const int rank = static_cast<int>(shape.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return base::kStatusCodeErrorInvalidParam;

  const auto index_type = inputs_[1]->getDataType();
  if (index_type != base::dataTypeOf<int32_t>() &&
      index_type != base::dataTypeOf<int64_t>()) return base::kStatusCodeErrorInvalidParam;
  const auto indices_shape = inputs_[1]->getShape();
  size_t count = base::shapeCount(indices_shape);
  size_t outer = base::shapeCount(shape, 0, axis);
  size_t block = base::shapeCount(shape, axis + 1) * inputs_[0]->getDataType().size();
  if (!outer || !count || !block) return base::kStatusCodeOk;
  const auto *src = static_cast<const char *>(inputs_[0]->getData());
  const void *indices = inputs_[1]->getData();
  auto *dst = static_cast<char *>(outputs_[0]->getData());
  if (!src || !indices || !dst) return base::kStatusCodeErrorNullParam;
  if (count > inputs_[1]->getSize() / index_type.size() ||
      outer > outputs_[0]->getSize() / block / count ||
      static_cast<size_t>(shape[axis]) > inputs_[0]->getSize() / block / outer)
    return base::kStatusCodeErrorInvalidValue;
  for (size_t i = 0; i < count; ++i) {
    int64_t index = index_type == base::dataTypeOf<int32_t>()
        ? static_cast<const int32_t *>(indices)[i]
        : static_cast<const int64_t *>(indices)[i];
    if (index < 0) index += shape[axis];
    if (index < 0 || index >= shape[axis]) return base::kStatusCodeErrorInvalidValue;
    for (size_t j = 0; j < outer; ++j)
      std::memcpy(dst + (j * count + i) * block,
                  src + (j * shape[axis] + index) * block, block);
  }
  return base::kStatusCodeOk;
}

base::Status OpGather::inferShape() {
  auto shape = inputs_[0]->getShape();
  const auto indices = inputs_[1]->getShape();
  auto param = static_cast<ir::GatherParam *>(op_desc_.op_param_.get());
  int axis = param ? param->axis_ : 0;
  const int rank = static_cast<int>(shape.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return base::kStatusCodeErrorInvalidParam;
  shape.erase(shape.begin() + axis);
  shape.insert(shape.begin() + axis, indices.begin(), indices.end());
  if (shape.empty()) {
    auto desc = outputs_[0]->getDesc();
    desc.shape_.clear();
    desc.stride_.clear();
    return outputs_[0]->justModify(desc) ? base::kStatusCodeOk : base::kStatusCodeErrorInvalidParam;
  }
  return outputs_[0]->reshape(shape);
}

base::Status gather(device::Tensor* input, device::Tensor* index,
                    std::shared_ptr<ir::GatherParam> param,
                    device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  std::unique_ptr<Op> op(createOp(input->getDeviceType(), "", ir::kOpTypeGather));
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }

  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(index, 1);
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

  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeGather, OpGather)

}  // namespace op
}  // namespace nndeploy
