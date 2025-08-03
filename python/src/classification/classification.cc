
#include "nndeploy/classification/classification.h"

#include "nndeploy/classification/result.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace classification {

NNDEPLOY_API_PYBIND11_MODULE("classification", m) {
  py::class_<ClassificationPostParam, base::Param,
             std::shared_ptr<ClassificationPostParam>>(
      m, "ClassificationPostParam")
      .def(py::init<>())
      .def_readwrite("topk_", &ClassificationPostParam::topk_)
      .def_readwrite("is_softmax_", &ClassificationPostParam::is_softmax_)
      .def_readwrite("version_", &ClassificationPostParam::version_);

  py::class_<ClassificationPostProcess, dag::Node>(m,
                                                   "ClassificationPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &ClassificationPostProcess::run);

  py::class_<ClassificationGraph, dag::Graph>(m, "ClassificationGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("defaultParam", &ClassificationGraph::defaultParam)
      .def("make", &ClassificationGraph::make)
      .def("setInferenceType", &ClassificationGraph::setInferenceType)
      .def("setInferParam", &ClassificationGraph::setInferParam)
      .def("setSrcPixelType", &ClassificationGraph::setSrcPixelType)
      .def("setTopk", &ClassificationGraph::setTopk)
      .def("setSoftmax", &ClassificationGraph::setSoftmax)
      .def("forward", &ClassificationGraph::forward,
           py::return_value_policy::reference);

  py::class_<ResnetGraph, dag::Graph>(m, "ResnetGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("defaultParam", &ResnetGraph::defaultParam)
      .def("make", &ResnetGraph::make)
      .def("setInferenceType", &ResnetGraph::setInferenceType)
      .def("setInferParam", &ResnetGraph::setInferParam)
      .def("setSrcPixelType", &ResnetGraph::setSrcPixelType)
      .def("setTopk", &ResnetGraph::setTopk)
      .def("setSoftmax", &ResnetGraph::setSoftmax)
      .def("forward", &ResnetGraph::forward,
           py::return_value_policy::reference);
}

}  // namespace classification
}  // namespace nndeploy
