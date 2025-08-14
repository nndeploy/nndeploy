#include "nndeploy/segment/segment_anything/sam.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace segment {

NNDEPLOY_API_PYBIND11_MODULE("segment", m) {
  // SAMPointsParam class binding
  py::class_<SAMPointsParam, base::Param, std::shared_ptr<SAMPointsParam>>(
      m, "SAMPointsParam")
      .def(py::init<>())
      .def_readwrite("points_", &SAMPointsParam::points_)
      .def_readwrite("labels_", &SAMPointsParam::labels_)
      .def_readwrite("ori_width", &SAMPointsParam::ori_width)
      .def_readwrite("ori_height", &SAMPointsParam::ori_height)
      .def_readwrite("version_", &SAMPointsParam::version_);

  // SelectPointNode class binding
  py::class_<SelectPointNode, dag::Node>(m, "SelectPointNode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("setPoints", &SelectPointNode::setPoints)
      .def("run", &SelectPointNode::run)
      .def("defaultParam", &SelectPointNode::defaultParam)
      .def_readwrite("points_", &SelectPointNode::points_)
      .def_readwrite("point_labels_", &SelectPointNode::point_labels_);

  // SAMGraph class binding
  py::class_<SAMGraph, dag::Graph>(m, "SAMGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("setInferParam", &SAMGraph::setInferParam)
      .def("defaultParam", &SAMGraph::defaultParam)
      .def("forward", &SAMGraph::forward,
           py::return_value_policy::reference);
}

}  // namespace segment
}  // namespace nndeploy