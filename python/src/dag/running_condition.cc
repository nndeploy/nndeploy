// #include "dag/dag.h"
// #include "nndeploy/base/param.h"
// #include "nndeploy/dag/edge.h"
// #include "nndeploy/dag/graph.h"
// #include "nndeploy_api_registry.h"

// namespace py = pybind11;
// namespace nndeploy {
// namespace dag {

// NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
//   py::class_<RunningCondition, Condition, PyRunningCondition<RunningCondition>>(
//       m, "RunningCondition", py::dynamic_attr())
//       .def(py::init<const std::string &>())
//       .def(py::init<const std::string &, std::vector<Edge *>,
//                     std::vector<Edge *>>())
//       .def("choose", &RunningCondition::choose);
// }

// }  // namespace dag
// }  // namespace nndeploy