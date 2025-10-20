#include "nndeploy/llm/stream_out.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace llm {

NNDEPLOY_API_PYBIND11_MODULE("llm", m) {
  py::class_<StreamOut, dag::Node>(m, "StreamOut")
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("init", &StreamOut::init)
      .def("run", &StreamOut::run);
}

}  // namespace llm
}  // namespace nndeploy
