#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<Condition, Graph, PyCondition<Condition>>(m, "Condition",
                                                py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>, std::vector<Edge *>>())
      .def("init", &Condition::init)
      .def("deinit", &Condition::deinit)
      .def("choose", &Condition::choose)
      .def("run", &Condition::run);
}

}
}