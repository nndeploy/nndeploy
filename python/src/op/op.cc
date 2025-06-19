#include "nndeploy_api_registry.h"
#include "op/op_func.h"

namespace nndeploy {
namespace op {
template <class OpBase = Op>
class PyOp : public OpBase {
 public:
  using OpBase::OpBase;

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, OpBase, "run", run);
  }

  base::Status inferDataType() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "infer_data_type",
                           inferDataType);
  }

  base::Status inferShape() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "infer_shape", inferShape);
  }

  base::Status inferDataFormat() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "infer_data_format",
                           inferDataFormat);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "deinit", deinit);
  }

  base::Status reshape(base::ShapeMap &shape_map) override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "reshape", reshape, shape_map);
  }

  base::Status preRun() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "pre_run", preRun);
  }

  uint64_t getWorkspaceSize() override {
    PYBIND11_OVERRIDE_NAME(uint64_t, OpBase, "get_workspace_size",
                           getWorkspaceSize);
  }

  void setWorkspace(void *workspace) override {
    PYBIND11_OVERRIDE_NAME(void, OpBase, "set_workspace", setWorkspace,
                           workspace);
  }

  uint64_t getFlops() override {
    PYBIND11_OVERRIDE_NAME(uint64_t, OpBase, "get_flops", getFlops);
  }

  base::Status checkOrAllocOutput() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "check_or_alloc_output",
                           checkOrAllocOutput);
  }

  base::Status postRun() override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "post_run", postRun);
  }

  base::Status setParam(std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "set_param", setParam, param);
  }

  std::shared_ptr<base::Param> getParam() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, OpBase, "get_param",
                           getParam);
  }

  base::Status setPrecisionType(base::PrecisionType precision_type) override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "set_precision_type",
                           setPrecisionType, precision_type);
  }

  base::Status setInput(device::Tensor *input) override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "set_input", setInput, input);
  }

  base::Status setOutput(device::Tensor *output) override {
    PYBIND11_OVERRIDE_NAME(base::Status, OpBase, "set_output", setOutput,
                           output);
  }
};

class PyOpCreator : public OpCreator {
 public:
  using OpCreator::OpCreator;

  Op *createOp(base::DeviceType device_type, const std::string &name,
               ir::OpType op_type, std::vector<std::string> &inputs,
               std::vector<std::string> &outputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(Op *, OpCreator, "create_op", createOp,
                                device_type, name, op_type, inputs, outputs);
  }

  std::shared_ptr<Op> createOpSharedPtr(
      base::DeviceType device_type, const std::string &name, ir::OpType op_type,
      std::vector<std::string> &inputs,
      std::vector<std::string> &outputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<Op>, OpCreator,
                                "create_op_shared_ptr", createOpSharedPtr,
                                device_type, name, op_type, inputs, outputs);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("op", m) {
  py::class_<Op, PyOp<Op>>(m, "Op", py::dynamic_attr())
      .def(py::init<>())
      .def("set_name", &Op::setName)
      .def("get_name", &Op::getName)
      .def("set_op_type", &Op::setOpType)
      .def("get_op_type", &Op::getOpType)
      .def("set_param", &Op::setParam)
      .def("get_param", &Op::getParam)
      .def("set_device_type", &Op::setDeviceType)
      .def("get_device_type", &Op::getDeviceType)
      .def("set_stream", &Op::setStream)
      .def("get_stream", &Op::getStream, py::return_value_policy::reference)
      .def("set_precision_type", &Op::setPrecisionType)
      .def("get_precision_type", &Op::getPrecisionType)
      .def("get_input_name", &Op::getInputName, py::arg("index") = 0)
      .def("get_output_name", &Op::getOutputName, py::arg("index") = 0)
      .def("get_input", &Op::getInput, py::arg("index") = 0,
           py::return_value_policy::reference)
      .def("get_output", &Op::getOutput, py::arg("index") = 0,
           py::return_value_policy::reference)
      .def("set_input", py::overload_cast<device::Tensor *>(&Op::setInput))
      .def("set_output", py::overload_cast<device::Tensor *>(&Op::setOutput))
      .def("set_input", py::overload_cast<device::Tensor *, int>(&Op::setInput))
      .def("set_output",
           py::overload_cast<device::Tensor *, int>(&Op::setOutput))
      .def("set_all_input_name",
           py::overload_cast<std::initializer_list<std::string>>(
               &Op::setAllInputName))
      .def("set_all_output_name",
           py::overload_cast<std::initializer_list<std::string>>(
               &Op::setAllOutputName))
      .def("set_all_input_name",
           py::overload_cast<std::vector<std::string> &>(&Op::setAllInputName))
      .def("set_all_output_name",
           py::overload_cast<std::vector<std::string> &>(&Op::setAllOutputName))
      .def("get_all_input_name", &Op::getAllInputName)
      .def("get_all_output_name", &Op::getAllOutputName)
      .def("get_all_input", &Op::getAllInput,
           py::return_value_policy::reference)
      .def("get_all_output", &Op::getAllOutput,
           py::return_value_policy::reference)
      .def("rm_input", &Op::rmInput)
      .def("set_all_input", &Op::setAllInput)
      .def("set_all_output", &Op::setAllOutput)
      .def("get_constructed", &Op::getConstructed)
      .def("set_parallel_type", &Op::setParallelType)
      .def("get_parallel_type", &Op::getParallelType)
      .def("set_inner_flag", &Op::setInnerFlag)
      .def("set_initialized_flag", &Op::setInitializedFlag)
      .def("get_initialized", &Op::getInitialized)
      .def("set_time_profile_flag", &Op::setTimeProfileFlag)
      .def("get_time_profile_flag", &Op::getTimeProfileFlag)
      .def("set_debug_flag", &Op::setDebugFlag)
      .def("get_debug_flag", &Op::getDebugFlag)
      .def("set_running_flag", &Op::setRunningFlag)
      .def("is_running", &Op::isRunning)
      .def("infer_data_type", &Op::inferDataType)
      .def("infer_shape", &Op::inferShape)
      .def("infer_data_format", &Op::inferDataFormat)
      .def("init", &Op::init)
      .def("deinit", &Op::deinit)
      .def("reshape", &Op::reshape)
      .def("pre_run", &Op::preRun)
      .def("get_workspace_size", &Op::getWorkspaceSize)
      .def("set_workspace", &Op::setWorkspace)
      .def("get_flops", &Op::getFlops)
      .def("check_or_alloc_output", &Op::checkOrAllocOutput)
      .def("run", &Op::run)
      .def("post_run", &Op::postRun);

  py::class_<OpCreator, PyOpCreator, std::shared_ptr<OpCreator>>(m, "OpCreator")
      .def(py::init<>())
      .def("create_op", &OpCreator::createOp,
           py::return_value_policy::take_ownership)
      .def("create_op_shared_ptr", &OpCreator::createOpSharedPtr,
           py::return_value_policy::take_ownership);

  m.def("register_op_creator",
        [](base::DeviceTypeCode device_type_code, ir::OpType op_type,
           std::shared_ptr<OpCreator> creator) {
          getGlobalOpCreatorMap()[device_type_code][op_type] = creator;
        });

  m.def(
      "create_op",
      py::overload_cast<base::DeviceType, const std::string &, ir::OpType,
                        std::vector<std::string> &, std::vector<std::string> &>(
          &createOp),
      py::arg("device_type"), py::arg("name"), py::arg("op_type"),
      py::arg("inputs"), py::arg("outputs"),
      py::return_value_policy::take_ownership);

  m.def("rms_norm", &rmsNormFunc, py::return_value_policy::take_ownership);
  m.def("batch_norm", &batchNormFunc, py::return_value_policy::take_ownership);
  m.def("relu", &reluFunc, py::return_value_policy::take_ownership);
  m.def("conv", &convFunc, py::return_value_policy::take_ownership);
  m.def("concat", &concatFunc, py::return_value_policy::take_ownership);
  m.def("add", &addFunc, py::return_value_policy::take_ownership);
  m.def("flatten", &flattenFunc, py::return_value_policy::take_ownership);
  m.def("gemm", &gemmFunc, py::return_value_policy::take_ownership);
  m.def("global_averagepool", &globalAveragepoolFunc,
        py::return_value_policy::take_ownership);
  m.def("maxpool", &maxPoolFunc, py::return_value_policy::take_ownership);
  m.def("mul", &mulFunc, py::return_value_policy::take_ownership);
  m.def("softmax", &softmaxFunc, py::return_value_policy::take_ownership);
  m.def("quantize_linear", &quantizeLinearFunc,
        py::return_value_policy::take_ownership);
  m.def("dequantize_linear", &dequantizeLinearFunc,
        py::return_value_policy::take_ownership);
  m.def("qlinear_conv", &qlinearConvFunc,
        py::return_value_policy::take_ownership);
}

}  // namespace op
}  // namespace nndeploy
