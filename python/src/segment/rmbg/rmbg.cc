#include "nndeploy/segment/rmbg/rmbg.h"

#include "nndeploy/segment/result.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace segment {

NNDEPLOY_API_PYBIND11_MODULE("segment", m) {
  py::class_<RMBGPostParam, base::Param, std::shared_ptr<RMBGPostParam>>(
      m, "RMBGPostParam")
      .def(py::init<>())
      .def_readwrite("version_", &RMBGPostParam::version_);

  py::class_<RMBGPostProcess, dag::Node>(m, "RMBGPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &RMBGPostProcess::run);

  py::class_<SegmentRMBGGraph, dag::Graph>(m, "SegmentRMBGGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("defaultParam", &SegmentRMBGGraph::defaultParam)
      .def("make", &SegmentRMBGGraph::make)
      .def("setInferParam", &SegmentRMBGGraph::setInferParam)
      .def("setSrcPixelType", &SegmentRMBGGraph::setSrcPixelType)
      .def("setVersion", &SegmentRMBGGraph::setVersion)
      .def("forward", &SegmentRMBGGraph::forward,
           py::return_value_policy::reference);
}

}  // namespace segment
}  // namespace nndeploy
