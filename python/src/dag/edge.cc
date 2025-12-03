#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "dag/dag.h"
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
      .def("set_queue_max_size", &Edge::setQueueMaxSize,
           py::arg("queue_max_size"))
      .def("get_queue_max_size", &Edge::getQueueMaxSize)
      .def("set_queue_overflow_policy", &Edge::setQueueOverflowPolicy,
           py::arg("policy"), py::arg("drop_count") = 1)
      .def("get_queue_overflow_policy", &Edge::getQueueOverflowPolicy)
      .def("get_queue_drop_count", &Edge::getQueueDropCount)

      // 并行类型设置和获取
      .def("set_parallel_type", &Edge::setParallelType, py::arg("paralle_type"))
      .def("get_parallel_type", &Edge::getParallelType)

      .def("empty", &Edge::empty)

      // 构造函数
      .def("construct", &Edge::construct)

      .def(
          "set",
          [](Edge& edge, py::object obj) {
            // NNDEPLOY_LOGE(
            //     "set obj type: %s.\n",
            //     obj.get_type().attr("__name__").cast<std::string>().c_str());
            auto* wrapper = new PyObjectWrapper(obj.ptr());
            // NNDEPLOY_LOGE("wrapper: %p", wrapper);
            // base::Status status = edge.set4py<PyObjectWrapper>(wrapper,
            // false); if (status != base::StatusCode::kStatusCodeOk) {
            //   throw std::runtime_error("Failed to set PyObjectWrapper");
            // }
            // return status;
            // 检查输入对象类型
            auto handle_status = [&](const base::Status& status,
                                     const char* error_msg) {
              if (status != base::StatusCode::kStatusCodeOk) {
                delete wrapper;
                throw std::runtime_error(error_msg);
              }
              return status;
            };
            if (py::isinstance<py::array>(obj)) {
              // numpy array类型
              auto buffer = obj.cast<py::array>();
              //        // 获取numpy数组的维度和形状
              py::buffer_info info = buffer.request();
              char kind = info.format.front();
              int channels = (info.ndim == 3) ? info.shape[2] : 1;
              int cv_depth;
#ifdef ENABLE_NNDEPLOY_OPENCV
              // 根据numpy数组的数据类型和每个元素的大小来确定OpenCV的数据类型
              switch (kind) {
                case 'B':
                  cv_depth = CV_8U;
                  break;
                case 'b':
                  cv_depth = CV_8S;
                  break;
                case 'H':
                  cv_depth = CV_16U;
                  break;
                case 'h':
                  cv_depth = CV_16S;
                  break;
                case 'i':
                  cv_depth = CV_32S;
                  break;
                case 'f':
                  cv_depth = CV_32F;
                  break;
                case 'd':
                  cv_depth = CV_64F;
                  break;
                default:
                  throw std::runtime_error("Unsupported data type kind: " +
                                           std::string(1, kind));
              }
              int type = CV_MAKETYPE(cv_depth, channels);
              cv::Mat* mat =
                  new cv::Mat(info.shape[0], info.shape[1], type, info.ptr);
              base::Status status;
              {
                py::gil_scoped_release release;
                status = edge.set4py(wrapper, mat, false);
              }
              return handle_status(status, "Failed to set cv::Mat");
#else
              NNDEPLOY_LOGE("set cv::Mat is not supported");
              delete wrapper;
              base::Status status =
                  base::StatusCode::kStatusCodeErrorNotSupport;
              return status;
#endif
            } else if (py::isinstance<device::Tensor>(obj)) {
              // Tensor类型
              auto* tensor = obj.cast<device::Tensor*>();
              base::Status status;
              {
                py::gil_scoped_release release;
                status = edge.set4py(wrapper, tensor);
              }
              return handle_status(status, "Failed to set device::Tensor");
            } else if (py::isinstance<device::Buffer>(obj)) {
              // Buffer类型
              auto* buffer = obj.cast<device::Buffer*>();
              base::Status status;
              {
                py::gil_scoped_release release;
                status = edge.set4py(wrapper, buffer);
              }
              return handle_status(status, "Failed to set device::Buffer");
            } else if (py::isinstance<base::Param>(obj)) {
              // Param类型
              auto param = obj.cast<std::shared_ptr<base::Param>>();
              base::Status status;
              {
                py::gil_scoped_release release;
                status = edge.set4py(wrapper, param.get());
              }
              return handle_status(status, "Failed to set base::Param");
            } else {
              base::Status status;
              {
                py::gil_scoped_release release;
                status = edge.set4py(wrapper, obj.ptr());
              }
              return handle_status(status, "Failed to set PyObject");
            }
          },
          py::arg("obj"))
      .def(
          "get",
          [](Edge& edge, const Node* node) {
            // 尝试获取不同类型的数据
            if (auto* tensor = edge.getTensor(node)) {
              py::gil_scoped_acquire acquire;
              return py::cast(tensor);
            } else if (auto* buffer = edge.getBuffer(node)) {
              py::gil_scoped_acquire acquire;
              return py::cast(buffer);
            } else if (auto* param = edge.getParam(node)) {
              py::gil_scoped_acquire acquire;
              return py::cast(param);
            }
#ifdef ENABLE_NNDEPLOY_OPENCV
            else if (auto* mat = edge.getCvMat(node)) {
              py::gil_scoped_acquire acquire;
              if (mat == nullptr) {
                return py::object(py::array());  // 返回空数组
              }

              std::string format;
              switch (mat->depth()) {
                case CV_8U:
                  format = "B";
                  break;
                case CV_8S:
                  format = "b";
                  break;
                case CV_16U:
                  format = "H";
                  break;
                case CV_16S:
                  format = "h";
                  break;
                case CV_32S:
                  format = "i";
                  break;
                case CV_32F:
                  format = "f";
                  break;
                case CV_64F:
                  format = "d";
                  break;
                default:
                  throw std::runtime_error("Unsupported cv::Mat data type");
              }

              std::vector<ssize_t> shape;
              std::vector<ssize_t> strides;

              if (mat->channels() == 1) {
                // 单通道图像
                shape = {static_cast<ssize_t>(mat->rows),
                         static_cast<ssize_t>(mat->cols)};
                strides = {static_cast<ssize_t>(mat->step[0]),
                           static_cast<ssize_t>(mat->elemSize1())};
              } else {
                // 多通道图像
                shape = {static_cast<ssize_t>(mat->rows),
                         static_cast<ssize_t>(mat->cols),
                         static_cast<ssize_t>(mat->channels())};
                strides = {
                    static_cast<ssize_t>(mat->step[0]),
                    static_cast<ssize_t>(mat->elemSize1() * mat->channels()),
                    static_cast<ssize_t>(mat->elemSize1())};
              }

              return py::object(
                  py::array(py::buffer_info(mat->data,         // 数据指针
                                            mat->elemSize1(),  // 单个元素大小
                                            format,            // 数据格式
                                            shape.size(),      // 维度数
                                            shape,             // 形状
                                            strides            // 步长
                                            )));
            }
#endif
            else if (auto* obj = edge.get<PyObject>(node)) {
              // 报错原因:
              // 1. 直接使用 obj->ob_type 访问 Python
              // 对象的类型不安全,可能导致段错误
              // 2. py::cast(obj) 无法正确处理 Python 对象的生命周期管理
              // 3. 缺少对 obj 是否为 nullptr 的检查
              // 正确的解决方案:
              if (!obj) {
                return py::object(py::none());
              }
              // 使用 PyObject_GetAttrString 安全地获取类型名称
              // PyObject* type_str =
              // PyObject_GetAttrString((PyObject*)Py_TYPE(obj), "__name__"); if
              // (type_str) {
              //   NNDEPLOY_LOGE("get obj type: %s",
              //   PyUnicode_AsUTF8(type_str));
              // }

              return py::reinterpret_borrow<py::object>(obj);
            }
            return py::object(py::none());
          },
          py::arg("node"), py::return_value_policy::reference,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_graph_output",
          [](Edge& edge) {
            // 尝试获取不同类型的数据
            if (auto* tensor = edge.getGraphOutputTensor()) {
              py::gil_scoped_acquire acquire;
              return py::cast(tensor);
            } else if (auto* buffer = edge.getGraphOutputBuffer()) {
              py::gil_scoped_acquire acquire;
              return py::cast(buffer);
            } else if (auto* param = edge.getGraphOutputParam()) {
              // NNDEPLOY_LOGE("get_graph_output param: %p", param);
              py::gil_scoped_acquire acquire;
              return py::cast(param);
            }
#ifdef ENABLE_NNDEPLOY_OPENCV
            else if (auto* mat = edge.getGraphOutputCvMat()) {
              py::gil_scoped_acquire acquire;
              if (mat == nullptr) {
                return py::object(py::array());  // 返回空数组
              }

              std::string format;
              switch (mat->depth()) {
                case CV_8U:
                  format = "B";
                  break;
                case CV_8S:
                  format = "b";
                  break;
                case CV_16U:
                  format = "H";
                  break;
                case CV_16S:
                  format = "h";
                  break;
                case CV_32S:
                  format = "i";
                  break;
                case CV_32F:
                  format = "f";
                  break;
                case CV_64F:
                  format = "d";
                  break;
                default:
                  throw std::runtime_error("Unsupported cv::Mat data type");
              }

              std::vector<ssize_t> shape;
              std::vector<ssize_t> strides;

              if (mat->channels() == 1) {
                // 单通道图像
                shape = {static_cast<ssize_t>(mat->rows),
                         static_cast<ssize_t>(mat->cols)};
                strides = {static_cast<ssize_t>(mat->step[0]),
                           static_cast<ssize_t>(mat->elemSize1())};
              } else {
                // 多通道图像
                shape = {static_cast<ssize_t>(mat->rows),
                         static_cast<ssize_t>(mat->cols),
                         static_cast<ssize_t>(mat->channels())};
                strides = {
                    static_cast<ssize_t>(mat->step[0]),
                    static_cast<ssize_t>(mat->elemSize1() * mat->channels()),
                    static_cast<ssize_t>(mat->elemSize1())};
              }

              return py::object(
                  py::array(py::buffer_info(mat->data,         // 数据指针
                                            mat->elemSize1(),  // 单个元素大小
                                            format,            // 数据格式
                                            shape.size(),      // 维度数
                                            shape,             // 形状
                                            strides            // 步长
                                            )));
            }
#endif
            else if (auto* obj = edge.getGraphOutput<PyObject>()) {
              py::gil_scoped_acquire acquire;
              // 报错原因:
              // 1. 直接使用 obj->ob_type 访问 Python
              // 对象的类型不安全,可能导致段错误
              // 2. py::cast(obj) 无法正确处理 Python 对象的生命周期管理
              // 3. 缺少对 obj 是否为 nullptr 的检查
              // 正确的解决方案:
              if (!obj) {
                return py::object(py::none());
              }
              // 使用 PyObject_GetAttrString 安全地获取类型名称
              // PyObject* type_str =
              // PyObject_GetAttrString((PyObject*)Py_TYPE(obj), "__name__"); if
              // (type_str) {
              //   NNDEPLOY_LOGE("get obj type: %s",
              //   PyUnicode_AsUTF8(type_str));
              // }
              return py::reinterpret_borrow<py::object>(obj);
            }
            return py::object(py::none());
          },
          py::return_value_policy::reference,
          py::call_guard<py::gil_scoped_release>())

      .def("create",
           static_cast<device::Buffer* (Edge::*)(device::Device*,
                                                 const device::BufferDesc&)>(
               &Edge::create),
           py::arg("device"), py::arg("desc"))
      .def("create",
           static_cast<device::Tensor* (Edge::*)(device::Device*,
                                                 const device::TensorDesc&,
                                                 std::string)>(&Edge::create),
           py::arg("device"), py::arg("desc"), py::arg("tensor_name") = "")
      .def("notify_written",
           static_cast<bool (Edge::*)(device::Buffer*)>(&Edge::notifyWritten),
           py::arg("buffer"))
      .def("notify_written",
           static_cast<bool (Edge::*)(device::Tensor*)>(&Edge::notifyWritten),
           py::arg("tensor"))

      // 索引和位置相关操作
      .def("get_index", &Edge::getIndex, py::arg("node"))
      .def("get_graph_output_index", &Edge::getGraphOutputIndex)
      .def("reset_index", &Edge::resetIndex)
      .def("get_position", &Edge::getPosition, py::arg("node"))
      .def("get_graph_output_position", &Edge::getGraphOutputPosition)

      // 更新和标记相关操作
      .def("update", &Edge::update, py::arg("node"),
           py::call_guard<py::gil_scoped_release>())
      .def("mark_graph_output", &Edge::markGraphOutput)

      // 生产者消费者相关操作
      .def("increase_producers", &Edge::increaseProducers, py::arg("producers"))
      .def("increase_consumers", &Edge::increaseConsumers, py::arg("consumers"))
      .def("get_producers", &Edge::getProducers)
      .def("get_consumers", &Edge::getConsumers)

      // 终止请求
      .def("request_terminate", &Edge::requestTerminate,
           py::call_guard<py::gil_scoped_release>())

      // 类型信息相关操作
      .def(
          "set_type",
          [](Edge& edge, py::object type_val) {
            if (py::isinstance<py::type>(type_val)) {
              py::type py_type = type_val.cast<py::type>();
              if (py_type.is(py::type::of<device::Buffer>())) {
                edge.setTypeInfo<device::Buffer>();
              } else if (py_type.is(py::type::of<device::Tensor>())) {
                edge.setTypeInfo<device::Tensor>();
              }
              // else if (py_type.is(py::type::of<base::Param>())) {
              //   edge.setTypeInfo<base::Param>();
              // }
              else {
                std::shared_ptr<EdgeTypeInfo> type_info =
                    std::make_shared<EdgeTypeInfo>();
                type_info->type_ = EdgeTypeFlag::kAny;
                // 获取类型的完整名称，包括模块路径
                py::object module = py_type.attr("__module__");
                std::string module_name = module.cast<std::string>();
                std::string type_name =
                    py_type.attr("__name__").cast<std::string>();
                type_info->type_name_ = module_name + "." + type_name;
                type_info->type_ptr_ = &typeid(py::type);
                type_info->type_holder_ =
                    std::make_shared<EdgeTypeInfo::TypeHolder<py::type>>();
                edge.setTypeInfo(type_info);
              }
            }
          },
          py::arg("type_val"))
      .def(
          "set_type_info",
          [](Edge& edge, std::shared_ptr<EdgeTypeInfo> type_info) {
            return edge.setTypeInfo(type_info);
          },
          py::arg("type_info"))
      .def("get_type_info", &Edge::getTypeInfo)
      .def("set_type_name", &Edge::setTypeName)
      .def("get_type_name", &Edge::getTypeName)
      .def(
          "check_type_info",
          [](Edge& edge, std::shared_ptr<EdgeTypeInfo> type_info) {
            return edge.checkTypeInfo(type_info);
          },
          py::arg("type_info"));
}

}  // namespace dag
}  // namespace nndeploy