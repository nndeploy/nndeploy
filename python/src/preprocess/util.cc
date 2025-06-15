
#include "nndeploy/preprocess/util.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  m.def("get_channel_by_pixel_type", &getChannelByPixelType,
        py::arg("pixel_type"));
}

}  // namespace preprocess
}  // namespace nndeploy