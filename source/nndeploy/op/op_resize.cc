
#include "nndeploy/op/op_resize.h"

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
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/util.h"

namespace nndeploy {
namespace op {

enum class KeepAspectRatioPolicy {
  STRETCH,
  NOT_LARGER,
  NOT_SMALLER,
};

void KeepAspectRatioHelper(KeepAspectRatioPolicy policy,
                           const base::IntVector& input_shape,
                           const std::vector<int>& axes,
                           std::vector<int>& sizes_data) {
  if (policy != KeepAspectRatioPolicy::NOT_LARGER &&
      policy != KeepAspectRatioPolicy::NOT_SMALLER) {
    return;
  }
  float scale = policy == KeepAspectRatioPolicy::NOT_LARGER
                    ? std::numeric_limits<float>::max()
                    : std::numeric_limits<float>::min();
  std::function<float(float, float)> reduce_f;
  if (policy == KeepAspectRatioPolicy::NOT_LARGER) {
    reduce_f = [](float a, float b) { return std::min(a, b); };
  } else {
    reduce_f = [](float a, float b) { return std::max(a, b); };
  }

  bool has_unknown_dim = false;
  for (size_t i = 0; i < sizes_data.size(); i++) {
    int d = axes.empty() ? i : axes[i];
    if (input_shape.size() <= d) {
      has_unknown_dim = true;
      break;
    }
    float s = sizes_data[i] / static_cast<float>(input_shape[d]);
    scale = reduce_f(scale, s);
  }
  // If there's at least one unknown dim we can't infer the output shape, since
  // it will depend on the original aspect ratio of the input.
  for (size_t i = 0; i < sizes_data.size(); i++) {
    int d = axes.empty() ? i : axes[i];
    sizes_data[i] = has_unknown_dim ? -1 : std::roundf(scale * input_shape[d]);
  }
}

void resizeShapeInferenceHelper(const base::IntVector& input_shape,
                                const std::vector<int>& sizes_data,
                                base::IntVector& output_shape) {
  if (!sizes_data.empty()) {
    for (int i = 0; i < input_shape.size(); ++i) {
      output_shape[i] = sizes_data[i];
    }
    return;
  }
}

void resizeShapeInferenceHelper(const base::IntVector& input_shape,
                                const std::vector<float>& scales_data,
                                base::IntVector& output_shape) {
  for (int i = 0; i < input_shape.size(); ++i) {
    // auto* dim = output_shape->mutable_dim(i);
    // If input_shape has dim_value, we calculate the scaled result
    // If input_shape doesn's have one, we leave it here
    if (input_shape.size() > i) {
      int dim_value = static_cast<int>(
          std::floor(static_cast<float>(input_shape[i]) * scales_data[i]));
      // If output_shape has dim_value, we validate the caculated result
      // If output_shape doesn's have one, we set it to the scaled result
      output_shape[i] = dim_value;
    }
  }
}

base::Status OpResize::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<ResizeParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");

  const auto& input_shape = inputs_[0]->getShape();
  auto output_shape = input_shape;  // 确定形状

  bool hasScalesInput = inputs_.size() > 2;
  bool hasSizesInput = inputs_.size() > 3;

  const float* scales =
      2 < inputs_.size() ? (const float*)inputs_[2]->getData() : nullptr;
  std::vector<int> sizes_data;
  if (3 < inputs_.size()) {
    bool found_sizes = false;
    const auto sizes_shape = inputs_[3]->getShape();
    if (sizes_shape.size() > 0) {
      found_sizes = true;
    }
    // If sizes is an empty shape, assume it's not provided
    if (found_sizes) {
      if (sizes_shape.size() == 0) {
        hasSizesInput = false;
      } else {
        for (int i = 0; i < sizes_shape.size(); ++i) {
          sizes_data.push_back(sizes_shape[i]);
        }
      }
    }
  }

  // If scales is an empty constant, assume it's not provided
  if (scales && scales == nullptr) {
    hasScalesInput = false;
    scales = nullptr;
  }

  int opset_version = 21;
  if (opset_version >= 13) {
    if (hasScalesInput + hasSizesInput != 1) {
      NNDEPLOY_LOGE(
          "Either `sizes` or `scales` must be provided, but not both of them");
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  auto keep_aspect_ratio_policy_attr = param->keep_aspect_ratio_policy_;
  KeepAspectRatioPolicy keep_aspect_ratio_policy =
      KeepAspectRatioPolicy::STRETCH;
  if (!keep_aspect_ratio_policy_attr.empty()) {
    auto str = keep_aspect_ratio_policy_attr;
    if (str == "stretch") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::STRETCH;
    } else if (str == "not_larger") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::NOT_LARGER;
    } else if (str == "not_smaller") {
      keep_aspect_ratio_policy = KeepAspectRatioPolicy::NOT_SMALLER;
    } else {
      NNDEPLOY_LOGE("Unknown value for `keep_aspect_ratio_policy`: %s.\n",
                    str.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  if (hasScalesInput &&
      keep_aspect_ratio_policy != KeepAspectRatioPolicy::STRETCH) {
    NNDEPLOY_LOGE(
        "Providing `scales` is incompatible with a `keep_aspect_ratio_policy` "
        "other than \"stretch\".");
    return base::kStatusCodeErrorInvalidParam;
  }

  auto axes_attr = param->axes_;
  size_t rank_x = input_shape.size();
  std::vector<int> axes;
  if (axes_attr != INT_MAX) {
    axes.push_back(axes_attr);
    bool flag = checkAxesRange(axes, rank_x);
    if (!flag) {
      return base::kStatusCodeErrorInvalidParam;
    }
    adjustNegativeAxes(axes, rank_x);
  }
  if (hasSizesInput) {
    if (!axes.empty()) {
      if (sizes_data.size() != axes.size()) {
        NNDEPLOY_LOGE(
            "Number of elements of input sizes(ld%) does not match the number "
            "of axes (ld%).\n",
            sizes_data.size(), axes.size());
      }
    } else {
      // sizes_data contains scales for all axes
      if (sizes_data.size() != rank_x) {
        NNDEPLOY_LOGE(
            "Number of elements of input 'sizes' (ld%) must be same as rank of "
            "input 'X' (ld%).\n",
            sizes_data.size(), rank_x);
      }
    }

    // Process sizes_data according to the selected policy
    KeepAspectRatioHelper(keep_aspect_ratio_policy, input_shape, axes,
                          sizes_data);

    // If axes subset is provided, populate new sizes_data with all dims
    if (!axes.empty()) {
      std::vector<int> tmp(rank_x);
      for (size_t i = 0; i < rank_x; i++) {
        tmp[i] = input_shape[i] ? input_shape[i] : -1;
      }
      for (size_t i = 0; i < axes.size(); i++) {
        int d = axes[i];
        tmp[d] = sizes_data[i];
      }
      std::swap(tmp, sizes_data);
    }

    resizeShapeInferenceHelper(input_shape, sizes_data, output_shape);
  } else if (nullptr != scales) {
    // Infer output shape's dimension value if 'scales' is known.
    if (inputs_[2]->getDataType() == base::dataTypeOf<float>()) {
      size_t size = inputs_[2]->getSize();
      std::vector<float> scales_data(size);
      for (size_t i = 0; i < size; i++) {
        scales_data[i] = scales[i];
      }

      if (!axes.empty()) {
        // scales_data contains scales for a subset of axes. The rest should not
        // be resized
        if (scales_data.size() != axes.size()) {
          NNDEPLOY_LOGE(
              "Number of elements of input 'scales' (ld%) does not match the "
              "number of axes (ld%).\n",
              scales_data.size(), axes.size());
        }

        std::vector<float> tmp(rank_x, 1.0f);
        for (size_t i = 0; i < axes.size(); i++) {
          int d = axes[i];
          tmp[d] = scales_data[i];
        }
        std::swap(tmp, scales_data);
      } else {
        // scales_data contains scales for all axes
        if (scales_data.size() != static_cast<size_t>(input_shape.size())) {
          NNDEPLOY_LOGE(
              "Number of elements of input 'scales' must be same as rank of "
              "input 'X'");
        }
      }
      resizeShapeInferenceHelper(input_shape, scales_data, output_shape);
    } else {
      NNDEPLOY_LOGE("Input 'scales' must have float element type.");
    }
  }  // nullptr != scales

  outputs_[0]->reshape(output_shape);

  return status;
}

}  // namespace op
}  // namespace nndeploy
