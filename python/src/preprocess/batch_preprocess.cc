#include "nndeploy/preprocess/batch_preprocess.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

template <typename Base = BatchPreprocess>
class PyBatchPreprocess : public Base {
 public:
  using Base::Base;  // 继承构造函数

  // base::Status setNodeKey(const std::string &key)  {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchPreprocess, "set_node_key",
  //                          setNodeKey, key);
  // }

  // base::Status setDataFormat(base::DataFormat data_format)  {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchPreprocess, "set_data_format",
  //                          setDataFormat, data_format);
  // }

  // base::DataFormat getDataFormat()  {
  //   PYBIND11_OVERRIDE_NAME(base::DataFormat, BatchPreprocess, "get_data_format",
  //                          getDataFormat);
  // }

  // base::Status setParam(base::Param *param)  {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchPreprocess, setParam, param);
  // }

  base::Status setParamSharedPtr(std::shared_ptr<base::Param> param)  {
    PYBIND11_OVERRIDE_NAME(base::Status, BatchPreprocess, "set_param",
                           setParamSharedPtr, param);
  }

  // base::Param *getParam()  {
  //   PYBIND11_OVERRIDE_NAME(base::Param*, BatchPreprocess, getParam);
  // }

  std::shared_ptr<base::Param> getParamSharedPtr()  {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, BatchPreprocess,
                           "get_param", getParamSharedPtr);
  }

  base::Status run()  {
    PYBIND11_OVERRIDE_NAME(base::Status, BatchPreprocess, "run", run);
  }

  // std::string serialize() {
  //   PYBIND11_OVERRIDE_NAME(std::string, BatchPreprocess, "serialize",
  //                          serialize);
  // }

  // base::Status deserialize(const std::string &json_str) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchPreprocess, "deserialize",
  //                          deserialize, json_str);
  // }
};

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<BatchPreprocess, PyBatchPreprocess<BatchPreprocess>,
             dag::CompositeNode>(
      m, "BatchPreprocess", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("set_node_key", &BatchPreprocess::setNodeKey)
      .def("set_data_format", &BatchPreprocess::setDataFormat)
      .def("get_data_format", &BatchPreprocess::getDataFormat)
      // .def("set_param", &BatchPreprocess::setParam)
      .def("set_param", &BatchPreprocess::setParamSharedPtr)
      // .def("get_param", &BatchPreprocess::getParam)
      .def("get_param", &BatchPreprocess::getParamSharedPtr)
      .def("run", &BatchPreprocess::run);
      // .def("serialize", py::overload_cast<rapidjson::Value &,
      //                        rapidjson::Document::AllocatorType &>(
      //          &BatchPreprocess::serialize),
      //      py::arg("json"), py::arg("allocator"));
      // .def("serialize", py::overload_cast<>(&BatchPreprocess::serialize))
      // .def("deserialize",
      //      py::overload_cast<rapidjson::Value &>(&BatchPreprocess::deserialize),
      //      py::arg("json"))
      // .def("deserialize",
      //      py::overload_cast<const std::string &>(&BatchPreprocess::deserialize),
      //      py::arg("json_str"));
}

}  // namespace preprocess
}  // namespace nndeploy