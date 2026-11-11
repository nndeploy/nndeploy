#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "dag/dag.h"
#include "numpy_array.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace dag {

// set4py transfers the wrapper (and an internally owned Mat) to the packet.
template <typename T>
static base::Status publish(Edge &edge, const py::object &owner, T *value,
                            bool external = true, bool written = true) {
  auto *wrapper = new PyObjectWrapper(owner.ptr());
  py::gil_scoped_release release;
  auto status = edge.set4py(wrapper, value, external, written);
  if (status != base::kStatusCodeOk) throw std::runtime_error(status.desc());
  return status;
}

base::Status setPythonEdgeData(Edge &edge, const py::object &obj) {
  if (py::isinstance<py::array>(obj)) {
#ifdef ENABLE_NNDEPLOY_OPENCV
    auto array = py::reinterpret_borrow<py::array>(obj);
    if (array.ndim() != 2 && array.ndim() != 3)
      throw py::value_error("expected a 2D image or a 3D HWC array");
    auto channels = array.ndim() == 3 ? array.shape(2) : 1;
    if (array.shape(0) > INT_MAX || array.shape(1) > INT_MAX ||
        channels < 1 || channels > CV_CN_MAX)
      throw py::value_error("invalid OpenCV image dimensions");
    auto dtype = array.dtype();
    int depth;
    if (dtype.equal(py::dtype::of<uint8_t>())) depth = CV_8U;
    else if (dtype.equal(py::dtype::of<int8_t>())) depth = CV_8S;
    else if (dtype.equal(py::dtype::of<uint16_t>())) depth = CV_16U;
    else if (dtype.equal(py::dtype::of<int16_t>())) depth = CV_16S;
    else if (dtype.equal(py::dtype::of<int32_t>())) depth = CV_32S;
    else if (dtype.equal(py::dtype::of<float>())) depth = CV_32F;
    else if (dtype.equal(py::dtype::of<double>())) depth = CV_64F;
    else throw py::type_error("unsupported OpenCV image dtype");
    auto storage = writableContiguousArray(array);
    auto *mat = new cv::Mat(static_cast<int>(array.shape(0)),
                            static_cast<int>(array.shape(1)),
                            CV_MAKETYPE(depth, static_cast<int>(channels)),
                            storage.mutable_data());
    return publish(edge, storage, mat, false);
#else
    return base::kStatusCodeErrorNotSupport;
#endif
  }
  if (py::isinstance<device::Tensor>(obj))
    return publish(edge, obj, obj.cast<device::Tensor *>());
  if (py::isinstance<device::Buffer>(obj))
    return publish(edge, obj, obj.cast<device::Buffer *>());
  if (py::isinstance<base::Param>(obj))
    return publish(edge, obj, obj.cast<base::Param *>());
  return publish(edge, obj, obj.ptr());
}

#ifdef ENABLE_NNDEPLOY_OPENCV
static py::object imageArray(cv::Mat *mat) {
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

py::object getPythonEdgeData(Edge &edge, const Node *node, bool graph_output) {
  std::shared_ptr<DataPacket> packet;
  {
    py::gil_scoped_release release;
    packet = edge.acquirePacket(node, graph_output);
  }
  if (!packet || packet->empty()) return py::none();
  // A Python submission already has an owning handle. For native Tensor and
  // Buffer outputs give the caller a shallow handle that survives packet reuse.
  switch (packet->getFlag()) {
#ifdef ENABLE_NNDEPLOY_OPENCV
    case EdgeTypeFlag::kCvMat:
      return imageArray(static_cast<cv::Mat *>(packet->getDataPtr()));
#endif
    case EdgeTypeFlag::kTensor: {
      auto *tensor = static_cast<device::Tensor *>(packet->getDataPtr());
      return packet->getWrapper() ? py::cast(tensor, py::return_value_policy::reference)
                                  : py::cast(device::Tensor(*tensor));
    }
    case EdgeTypeFlag::kBuffer: {
      auto *buffer = static_cast<device::Buffer *>(packet->getDataPtr());
      return packet->getWrapper() ? py::cast(buffer, py::return_value_policy::reference)
                                  : py::cast(device::Buffer(*buffer));
    }
    case EdgeTypeFlag::kParam:
      return py::cast(std::shared_ptr<base::Param>(packet, static_cast<base::Param *>(packet->getDataPtr())));
    default:
      if (auto *obj = packet->get<PyObject>())
        return py::reinterpret_borrow<py::object>(obj);
      return py::none();
  }
}

template <typename T, typename... Args>
static py::object createPythonEdgeData(Edge &edge, Args&&... args) {
  auto value = std::make_unique<T>(std::forward<Args>(args)...);
  auto *ptr = value.get();
  auto owner = py::cast(std::move(value));
  publish(edge, owner, ptr, true, false);
  return owner;
}

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

      .def("set", &setPythonEdgeData, py::arg("obj"))
      .def("get", [](Edge &edge, const Node *node) {
        return getPythonEdgeData(edge, node, false);
      }, py::arg("node"))
      .def("get_graph_output", [](Edge &edge) {
        return getPythonEdgeData(edge, nullptr, true);
      })
      .def("create", [](Edge &edge, device::Device *device, const device::BufferDesc &desc) {
        return createPythonEdgeData<device::Buffer>(edge, device, desc);
      }, py::arg("device"), py::arg("desc"))
      .def("create", [](Edge &edge, device::Device *device, const device::TensorDesc &desc,
                        const std::string &name) {
        return createPythonEdgeData<device::Tensor>(edge, device, desc, name.empty() ? edge.getName() : name);
      }, py::arg("device"), py::arg("desc"), py::arg("tensor_name") = "")
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