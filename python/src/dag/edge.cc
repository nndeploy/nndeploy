#include "nndeploy/dag/edge.h"

#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<Edge>(m, "Edge", py::dynamic_attr())
      // 构造函数
      .def(py::init<>())
      .def(py::init<const std::string&>())

      // 基本信息获取
      .def("get_name", &Edge::getName)

      // 并行类型设置和获取
      .def("set_parallel_type", &Edge::setParallelType, py::arg("paralle_type"))
      .def("get_parallel_type", &Edge::getParallelType)

      // 构造函数
      .def("construct", &Edge::construct)

      // Buffer相关操作
      .def("set", py::overload_cast<device::Buffer*, int, bool>(&Edge::set),
           py::arg("buffer"), py::arg("index"), py::arg("is_external") = true)
      .def("set", py::overload_cast<device::Buffer&, int>(&Edge::set),
           py::arg("buffer"), py::arg("index"))
      .def("create",
           static_cast<device::Buffer* (
               Edge::*)(device::Device*, const device::BufferDesc&, int)>(
               &Edge::create),
           py::arg("device"), py::arg("desc"), py::arg("index"),
           py::return_value_policy::reference)
      .def("notify_written",
           py::overload_cast<device::Buffer*>(&Edge::notifyWritten),
           py::arg("buffer"))
      .def("get_buffer", &Edge::getBuffer, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_buffer", &Edge::getGraphOutputBuffer,
           py::return_value_policy::reference)

#ifdef ENABLE_NNDEPLOY_OPENCV
      // OpenCV Mat相关操作
      .def("set", py::overload_cast<cv::Mat*, int, bool>(&Edge::set),
           py::arg("cv_mat"), py::arg("index"), py::arg("is_external") = true)
      .def("set", py::overload_cast<cv::Mat&, int>(&Edge::set),
           py::arg("cv_mat"), py::arg("index"))
      .def(
          "create",
          static_cast<cv::Mat* (Edge::*)(int, int, int, const cv::Vec3b&, int)>(
              &Edge::create),
          py::arg("rows"), py::arg("cols"), py::arg("type"), py::arg("value"),
          py::arg("index"), py::return_value_policy::reference)
      .def("notify_written", py::overload_cast<cv::Mat*>(&Edge::notifyWritten),
           py::arg("cv_mat"))
      .def("get_cv_mat", &Edge::getCvMat, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_cv_mat", &Edge::getGraphOutputCvMat,
           py::return_value_policy::reference)
#endif

      // Tensor相关操作
      .def("set", py::overload_cast<device::Tensor*, int, bool>(&Edge::set),
           py::arg("tensor"), py::arg("index"), py::arg("is_external") = true)
      .def("set", py::overload_cast<device::Tensor&, int>(&Edge::set),
           py::arg("tensor"), py::arg("index"))
      .def("create",
           static_cast<device::Tensor* (
               Edge::*)(device::Device*, const device::TensorDesc&, int)>(
               &Edge::create),
           py::arg("device"), py::arg("desc"), py::arg("index"),
           py::return_value_policy::reference)
      .def("notify_written",
           py::overload_cast<device::Tensor*>(&Edge::notifyWritten),
           py::arg("tensor"))
      .def("get_tensor", &Edge::getTensor, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_tensor", &Edge::getGraphOutputTensor,
           py::return_value_policy::reference)

      // Param相关操作
      .def("set", py::overload_cast<base::Param*, int, bool>(&Edge::set),
           py::arg("param"), py::arg("index"), py::arg("is_external") = true)
      .def("set", py::overload_cast<base::Param&, int>(&Edge::set),
           py::arg("param"), py::arg("index"))
      .def("notify_written",
           py::overload_cast<base::Param*>(&Edge::notifyWritten),
           py::arg("param"))
      .def("get_param", &Edge::getParam, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_param", &Edge::getGraphOutputParam,
           py::return_value_policy::reference)

      // 索引和位置相关操作
      .def("get_index", &Edge::getIndex, py::arg("node"))
      .def("get_graph_output_index", &Edge::getGraphOutputIndex)
      .def("get_position", &Edge::getPosition, py::arg("node"))
      .def("get_graph_output_position", &Edge::getGraphOutputPosition)

      // 更新和标记相关操作
      .def("update", &Edge::update, py::arg("node"))
      .def("mark_graph_output", &Edge::markGraphOutput)

      // 生产者消费者相关操作
      .def("increase_producers", &Edge::increaseProducers, py::arg("producers"))
      .def("increase_consumers", &Edge::increaseConsumers, py::arg("consumers"))

      // 终止请求
      .def("request_terminate", &Edge::requestTerminate);

}

}  // namespace dag
}  // namespace nndeploy