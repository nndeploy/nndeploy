
#include "nndeploy/preprocess/util.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  m.def("get_channel_by_pixel_type", &getChannelByPixelType,
        py::arg("pixel_type"),
        "根据像素类型获取通道数\n"
        "Args:\n"
        "    pixel_type (base.PixelType): 像素类型\n"
        "Returns:\n"
        "    int: 通道数");
}

}  // namespace dag
}  // namespace nndeploy