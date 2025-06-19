
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

  py::class_<ClassificationResnetGraph, dag::Graph>(m,
                                                    "ClassificationResnetGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("defaultParam", &ClassificationResnetGraph::defaultParam)
      .def("make", &ClassificationResnetGraph::make)
      .def("setInferenceType", &ClassificationResnetGraph::setInferenceType)
      .def("setInferParam", &ClassificationResnetGraph::setInferParam)
      .def("setSrcPixelType", &ClassificationResnetGraph::setSrcPixelType)
      .def("setTopk", &ClassificationResnetGraph::setTopk)
      .def("setSoftmax", &ClassificationResnetGraph::setSoftmax)
      .def("forward", &ClassificationResnetGraph::forward,
           py::return_value_policy::reference);
}

}  // namespace classification
}  // namespace nndeploy
