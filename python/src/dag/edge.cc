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

      // 队列最大值设置和获取
      .def("set_queue_max_size", &Edge::setQueueMaxSize, py::arg("queue_max_size"))
      .def("get_queue_max_size", &Edge::getQueueMaxSize)

      // 并行类型设置和获取
      .def("set_parallel_type", &Edge::setParallelType, py::arg("paralle_type"))
      .def("get_parallel_type", &Edge::getParallelType)

      // 构造函数
      .def("construct", &Edge::construct)

      // Buffer相关操作
      .def("set", py::overload_cast<device::Buffer*, bool>(&Edge::set),
           py::arg("buffer"), py::arg("is_external") = true)
      .def("set", py::overload_cast<device::Buffer&>(&Edge::set),
           py::arg("buffer"))
      .def("create",
           static_cast<device::Buffer* (Edge::*)(device::Device*,
                                                 const device::BufferDesc&)>(
               &Edge::create),
           py::arg("device"), py::arg("desc"),
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
      .def("set", py::overload_cast<cv::Mat*, bool>(&Edge::set),
           py::arg("cv_mat"), py::arg("is_external") = true)
      .def("set", py::overload_cast<cv::Mat&>(&Edge::set), py::arg("cv_mat"))
      .def("create",
           static_cast<cv::Mat* (Edge::*)(int, int, int, const cv::Vec3b&)>(
               &Edge::create),
           py::arg("rows"), py::arg("cols"), py::arg("type"), py::arg("value"),
           py::return_value_policy::reference)
      .def("notify_written", py::overload_cast<cv::Mat*>(&Edge::notifyWritten),
           py::arg("cv_mat"))
      .def("get_cv_mat", &Edge::getCvMat, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_cv_mat", &Edge::getGraphOutputCvMat,
           py::return_value_policy::reference)
#endif

      // Tensor相关操作
      .def("set", py::overload_cast<device::Tensor*, bool>(&Edge::set),
           py::arg("tensor"), py::arg("is_external") = true)
      .def("set", py::overload_cast<device::Tensor&>(&Edge::set),
           py::arg("tensor"))
      .def("create",
           static_cast<device::Tensor* (Edge::*)(device::Device*,
                                                 const device::TensorDesc&,
                                                 std::string)>(&Edge::create),
           py::arg("device"), py::arg("desc"), py::arg("tensor_name") = "",
           py::return_value_policy::reference)
      .def("notify_written",
           py::overload_cast<device::Tensor*>(&Edge::notifyWritten),
           py::arg("tensor"))
      .def("get_tensor", &Edge::getTensor, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_tensor", &Edge::getGraphOutputTensor,
           py::return_value_policy::reference)

      // Param相关操作
      .def("set", py::overload_cast<base::Param*, bool>(&Edge::set),
           py::arg("param"), py::arg("is_external") = true)
      .def("set", py::overload_cast<base::Param&>(&Edge::set), py::arg("param"))
      .def("notify_written",
           py::overload_cast<base::Param*>(&Edge::notifyWritten),
           py::arg("param"))
      .def("get_param", &Edge::getParam, py::arg("node"),
           py::return_value_policy::reference)
      .def("get_graph_output_param", &Edge::getGraphOutputParam,
           py::return_value_policy::reference)

      // 任意类型相关操作
      // 任意类型相关操作
      .def("set_any", [](Edge& edge, py::object obj) {
          // 创建一个PyObjectWrapper来存储Python对象
          // PyObjectWrapper 结构体用于包装和管理Python对象的生命周期
          struct PyObjectWrapper {
              PyObject* obj;  // 指向Python对象的指针
              
              // 构造函数:接收一个Python对象指针,增加引用计数防止对象被销毁
              PyObjectWrapper(PyObject* o) : obj(o) { Py_INCREF(obj); }
              
              // 析构函数:减少引用计数,允许Python回收对象
              ~PyObjectWrapper() { Py_DECREF(obj); }
          };
          
          // 创建包装器并增加引用计数
          auto* wrapper = new PyObjectWrapper(obj.ptr());
          
          // 设置类型信息
          auto type_info = std::make_shared<EdgeTypeInfo>();
          type_info->setType<PyObjectWrapper>();
          type_info->setTypeName(
              py::str(obj.get_type().attr("__module__")).cast<std::string>() + "." +
              py::str(obj.get_type().attr("__name__")).cast<std::string>()
          );
          edge.setTypeInfo(type_info);
          
          // 将包装器传递给Edge
          bool is_external = false;
          return edge.setAny<PyObjectWrapper>(wrapper, is_external);
      }, py::arg("obj"))

      .def("create_any", [](Edge& edge, py::object type_obj, py::args args, py::kwargs kwargs) {
          // 使用Python的type对象创建新实例
          py::object instance = type_obj(*args, **kwargs);
          
          // 包装并存储新创建的对象
          struct PyObjectWrapper {
              PyObject* obj;
              PyObjectWrapper(PyObject* o) : obj(o) { Py_INCREF(obj); }
              ~PyObjectWrapper() { Py_DECREF(obj); }
          };
          
          auto* wrapper = new PyObjectWrapper(instance.ptr());
          
          // 设置类型信息
          auto type_info = std::make_shared<EdgeTypeInfo>();
          type_info->setType<PyObjectWrapper>();
          type_info->setTypeName(
              py::str(type_obj.attr("__module__")).cast<std::string>() + "." +
              py::str(type_obj.attr("__name__")).cast<std::string>()
          );
          edge.setTypeInfo(type_info);

          edge.setAny<PyObjectWrapper>(wrapper, false);
          
          return instance;
      })

      .def("notify_any_written", [](Edge& edge, py::object obj) {
          struct PyObjectWrapper {
              PyObject* obj;
              PyObjectWrapper(PyObject* o) : obj(o) { Py_INCREF(obj); }
              ~PyObjectWrapper() { Py_DECREF(obj); }
          };
          
          auto* wrapper = new PyObjectWrapper(obj.ptr());
          return edge.notifyAnyWritten<PyObjectWrapper>(wrapper);
      }, py::arg("obj"))

      .def("get_any", [](Edge& edge, const Node* node) {
          struct PyObjectWrapper {
              PyObject* obj;
              PyObjectWrapper(PyObject* o) : obj(o) { Py_INCREF(obj); }
              ~PyObjectWrapper() { Py_DECREF(obj); }
          };
          
          auto* wrapper = edge.getAny<PyObjectWrapper>(node);
          if (!wrapper) {
              return py::object(py::none());  // 显式转换为 py::object
          }
          return py::reinterpret_borrow<py::object>(wrapper->obj);
      }, py::arg("node"))

      .def("get_graph_output_any", [](Edge& edge) {
          struct PyObjectWrapper {
              PyObject* obj;
              PyObjectWrapper(PyObject* o) : obj(o) { Py_INCREF(obj); }
              ~PyObjectWrapper() { Py_DECREF(obj); }
          };
          
          auto* wrapper = edge.getGraphOutputAny<PyObjectWrapper>();
          if (!wrapper) {
              return py::object(py::none());  // 显式转换为 py::object
          }
          return py::reinterpret_borrow<py::object>(wrapper->obj);
      })


      // 索引和位置相关操作
      .def("get_index", &Edge::getIndex, py::arg("node"))
      .def("get_graph_output_index", &Edge::getGraphOutputIndex)
      .def("reset_index", &Edge::resetIndex)
      .def("get_position", &Edge::getPosition, py::arg("node"))
      .def("get_graph_output_position", &Edge::getGraphOutputPosition)

      // 更新和标记相关操作
      .def("update", &Edge::update, py::arg("node"))
      .def("mark_graph_output", &Edge::markGraphOutput)

      // 生产者消费者相关操作
      .def("increase_producers", &Edge::increaseProducers, py::arg("producers"))
      .def("increase_consumers", &Edge::increaseConsumers, py::arg("consumers"))

      // 终止请求
      .def("request_terminate", &Edge::requestTerminate)

      // 类型信息相关操作
      .def(
          "set_type_info",
          [](Edge& edge, std::shared_ptr<EdgeTypeInfo> type_info) {
            return edge.setTypeInfo(type_info);
          },
          py::arg("type_info"))
      .def("get_type_info", &Edge::getTypeInfo)
      .def(
          "check_type_info",
          [](Edge& edge, std::shared_ptr<EdgeTypeInfo> type_info) {
            return edge.checkTypeInfo(type_info);
          },
          py::arg("type_info"));
}

}  // namespace dag
}  // namespace nndeploy