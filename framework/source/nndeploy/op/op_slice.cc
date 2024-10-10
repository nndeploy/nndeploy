
#include "nndeploy/op/op_slice.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/util.h"

namespace nndeploy {
namespace op {

base::Status OpSlice::inferShape() {
  base::Status status = base::kStatusCodeOk;
  const auto& input_shape = inputs_[0]->getShape();
  std::vector<int64_t> starts;
  if (inputs_[1]->getDataType() == base::dataTypeOf<int32_t>()) {
    int32_t* input_data = static_cast<int32_t*>(inputs_[1]->getData());
    size_t input_size = inputs_[1]->getSize() / sizeof(int32_t);
    starts.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      starts[i] = static_cast<int64_t>(input_data[i]);
    }
  } else {
    int64_t* input_data = static_cast<int64_t*>(inputs_[1]->getData());
    size_t input_size = inputs_[1]->getSize() / sizeof(int64_t);
    starts.resize(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      starts[i] = input_data[i];
    }
  }
  std::vector<int64_t> ends(starts.size());
  if (inputs_[2]->getDataType() == base::dataTypeOf<int32_t>()) {
    int32_t* input_data = static_cast<int32_t*>(inputs_[2]->getData());
    size_t input_size = inputs_[2]->getSize() / sizeof(int32_t);
    for (size_t i = 0; i < input_size; ++i) {
      ends[i] = static_cast<int64_t>(input_data[i]);
    }
  } else {
    int64_t* input_data = static_cast<int64_t*>(inputs_[2]->getData());
    size_t input_size = inputs_[2]->getSize() / sizeof(int64_t);
    for (size_t i = 0; i < input_size; ++i) {
      ends[i] = input_data[i];
    }
  }
  std::vector<int64_t> axes(starts.size());
  if (inputs_[3]->getDataType() == base::dataTypeOf<int32_t>()) {
    int32_t* input_data = static_cast<int32_t*>(inputs_[3]->getData());
    size_t input_size = inputs_[3]->getSize() / sizeof(int32_t);
    for (size_t i = 0; i < input_size; ++i) {
      axes[i] = static_cast<int64_t>(input_data[i]);
    }
  } else {
    int64_t* input_data = static_cast<int64_t*>(inputs_[3]->getData());
    size_t input_size = inputs_[3]->getSize() / sizeof(int64_t);
    for (size_t i = 0; i < input_size; ++i) {
      axes[i] = input_data[i];
    }
  }
  std::vector<int64_t> steps(starts.size(), 1);
  if (inputs_.size() == 5) {
    if (inputs_[4]->getDataType() == base::dataTypeOf<int32_t>()) {
      int32_t* input_data = static_cast<int32_t*>(inputs_[4]->getData());
      size_t input_size = inputs_[4]->getSize() / sizeof(int32_t);
      for (size_t i = 0; i < input_size; ++i) {
        steps[i] = static_cast<int64_t>(input_data[i]);
      }
    } else {
      int64_t* input_data = static_cast<int64_t*>(inputs_[4]->getData());
      size_t input_size = inputs_[4]->getSize() / sizeof(int64_t);
      for (size_t i = 0; i < input_size; ++i) {
        steps[i] = input_data[i];
      }
    }
  }

  base::IntVector output_shape = input_shape;

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    int64_t start = starts[i];
    int64_t end = ends[i];
    int64_t step = steps[i];

    if (start < 0) {
      start += input_shape[axis];
    }
    if (end < 0) {
      end += input_shape[axis];
    }

    start = std::max(int64_t(0), std::min(start, (int64_t)input_shape[axis]));
    end = std::max(int64_t(0), std::min(end, (int64_t)input_shape[axis]));

    output_shape[axis] = (end - start + step - 1) / step;
  }

  outputs_[0]->reshape(output_shape);

  return status;
}

}  // namespace op
}  // namespace nndeploy
