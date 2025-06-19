
#include "nndeploy/classification/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace classification {

NNDEPLOY_API_PYBIND11_MODULE("classification", m) {
  py::class_<ClassificationLableResult, base::Param,
             std::shared_ptr<ClassificationLableResult>>(
      m, "ClassificationLableResult")
      .def(py::init<>())
      .def_readwrite("index", &ClassificationLableResult::index_)
      .def_readwrite("label_ids", &ClassificationLableResult::label_ids_)
      .def_readwrite("scores_", &ClassificationLableResult::scores_)
      .def_readwrite("feature_", &ClassificationLableResult::feature_);

  py::class_<ClassificationResult, base::Param,
             std::shared_ptr<ClassificationResult>>(m, "ClassificationResult")
      .def(py::init<>())
      .def_property(
          "labels_",
          [](ClassificationResult& self) { return py::cast(self.labels_); },
          [](ClassificationResult& self, py::list list) {
            self.labels_.clear();
            for (auto item : list) {
              self.labels_.push_back(item.cast<ClassificationLableResult>());
            }
          });
}

}  // namespace detect
}  // namespace nndeploy
