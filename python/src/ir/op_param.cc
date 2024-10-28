#include "nndeploy/ir/op_param.h"

#include <pybind11/stl.h>

#include <vector>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace ir {

class MyClass {
 public:
  std::vector<int> contents;
};

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<MyClass>(m, "MyClass")
      .def(py::init<>())
      .def_readwrite("contents", &MyClass::contents);

  // 导出 OpParam 类
  py::class_<OpParam>(m, "OpParam")
      .def(py::init<>())
      .def_readwrite("reserved_", &OpParam::reserved_);

  // 导出 BatchNormalizationParam 类
  py::class_<BatchNormalizationParam, OpParam>(m, "BatchNormalizationParam")
      .def(py::init<>())
      .def_readwrite("epsilon_", &BatchNormalizationParam::epsilon_)
      .def_readwrite("momentum_", &BatchNormalizationParam::momentum_)
      .def_readwrite("training_mode_",
                     &BatchNormalizationParam::training_mode_);

  // 导出 ConcatParam 类
  py::class_<ConcatParam, OpParam>(m, "ConcatParam")
      .def(py::init<>())
      .def_readwrite("axis_", &ConcatParam::axis_);

  // 导出 ConvParam 类
  py::class_<ConvParam, OpParam>(m, "ConvParam")
      .def(py::init<>())
      .def_readwrite("auto_pad_", &ConvParam::auto_pad_)
      .def_readwrite("dilations_", &ConvParam::dilations_)
      .def_readwrite("group_", &ConvParam::group_)
      .def_readwrite("kernel_shape_", &ConvParam::kernel_shape_)
      .def_readwrite("pads_", &ConvParam::pads_)
      .def_readwrite("strides_", &ConvParam::strides_);

  // 导出 MaxPoolParam 类
  py::class_<MaxPoolParam, OpParam>(m, "MaxPoolParam")
      .def(py::init<>())
      .def_readwrite("auto_pad_", &MaxPoolParam::auto_pad_)
      .def_readwrite("ceil_mode_", &MaxPoolParam::ceil_mode_)
      .def_readwrite("dilations_", &MaxPoolParam::dilations_)
      .def_readwrite("kernel_shape_", &MaxPoolParam::kernel_shape_)
      .def_readwrite("pads_", &MaxPoolParam::pads_)
      .def_readwrite("storage_order_", &MaxPoolParam::storage_order_)
      .def_readwrite("strides_", &MaxPoolParam::strides_);

  // 导出 ReshapeParam 类
  py::class_<ReshapeParam, OpParam>(m, "ReshapeParam")
      .def(py::init<>())
      .def_readwrite("allowzero_", &ReshapeParam::allowzero_);

  // 导出 ResizeParam 类
  py::class_<ResizeParam, OpParam>(m, "ResizeParam")
      .def(py::init<>())
      .def_readwrite("antialias_", &ResizeParam::antialias_)
      .def_readwrite("axes_", &ResizeParam::axes_)
      .def_readwrite("coordinate_transformation_mode_",
                     &ResizeParam::coordinate_transformation_mode_)
      .def_readwrite("cubic_coeff_a_", &ResizeParam::cubic_coeff_a_)
      .def_readwrite("exclude_outside_", &ResizeParam::exclude_outside_)
      .def_readwrite("extrapolation_value_", &ResizeParam::extrapolation_value_)
      .def_readwrite("keep_aspect_ratio_policy_",
                     &ResizeParam::keep_aspect_ratio_policy_)
      .def_readwrite("mode_", &ResizeParam::mode_)
      .def_readwrite("nearest_mode_", &ResizeParam::nearest_mode_);

  // 导出 SoftmaxParam 类
  py::class_<SoftmaxParam, OpParam>(m, "SoftmaxParam")
      .def(py::init<>())
      .def_readwrite("axis_", &SoftmaxParam::axis_);

  // 导出 SplitParam 类
  py::class_<SplitParam, OpParam>(m, "SplitParam")
      .def(py::init<>())
      .def_readwrite("axis_", &SplitParam::axis_)
      .def_readwrite("num_outputs_", &SplitParam::num_outputs_);

  // 导出 TransposeParam 类
  py::class_<TransposeParam, OpParam>(m, "TransposeParam")
      .def(py::init<>())
      .def_readwrite("perm_", &TransposeParam::perm_);

  // 导出 RMSNormParam 类
  py::class_<RMSNormParam, OpParam>(m, "RMSNormParam")
      .def(py::init<>())
      .def_readwrite("eps_", &RMSNormParam::eps_)
      .def_readwrite("is_last_", &RMSNormParam::is_last_);
}
}  // namespace ir
}  // namespace nndeploy