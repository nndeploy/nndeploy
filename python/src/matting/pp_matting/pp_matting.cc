#include "nndeploy/matting/pp_matting/pp_matting.h"

#include "nndeploy/matting/result.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace matting {

NNDEPLOY_API_PYBIND11_MODULE("matting", m) {
  py::class_<PPMattingPostParam, base::Param, std::shared_ptr<PPMattingPostParam>>(
      m, "PPMattingPostParam")
      .def(py::init<>())
      .def_readwrite("alpha_h_", &PPMattingPostParam::alpha_h_)
      .def_readwrite("alpha_w_", &PPMattingPostParam::alpha_w_)
      .def_readwrite("output_h_", &PPMattingPostParam::output_h_)
      .def_readwrite("output_w_", &PPMattingPostParam::output_w_);

  py::class_<PPMattingPostProcess, dag::Node>(m, "PPMattingPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &PPMattingPostProcess::run);

  py::class_<PPMattingGraph, dag::Graph>(m, "PPMattingGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("default_param", &PPMattingGraph::defaultParam)
      .def("make", &PPMattingGraph::make)
      .def("set_infer_param", &PPMattingGraph::setInferParam)
      .def("set_model_hw", &PPMattingGraph::setModelHW);
}

}  // namespace matting
}  // namespace nndeploy
