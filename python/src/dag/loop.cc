// #include "dag/dag.h"
// #include "nndeploy/base/param.h"
// #include "nndeploy/dag/edge.h"
// #include "nndeploy/dag/graph.h"
// #include "nndeploy_api_registry.h"

// namespace py = pybind11;
// namespace nndeploy {
// namespace dag {

// NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
//   py::class_<Loop, Graph, PyLoop<Loop>>(m, "Loop", py::dynamic_attr())
//       .def(py::init<const std::string &>())
//       .def(py::init<const std::string &, std::vector<Edge *>,
//                     std::vector<Edge *>>())
//       .def("init", &Loop::init)
//       .def("deinit", &Loop::deinit)
//       .def("loops", &Loop::loops)
//       .def("run", &Loop::run);
// }

// }  // namespace dag
// }  // namespace nndeploy