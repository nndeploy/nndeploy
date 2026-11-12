#include <limits>

#include "dag/dag.h"
#include "nndeploy/dag/state.h"

namespace nndeploy {
namespace dag {

namespace {
template <typename T>
struct PythonOwner {
  std::shared_ptr<PyObjectWrapper> owner;
  void operator()(T*) const {}  // The Python wrapper owns the native object.
};

template <typename T>
std::shared_ptr<T> retainNative(py::handle object) {
  return std::shared_ptr<T>(object.cast<T*>(),
      PythonOwner<T>{std::make_shared<PyObjectWrapper>(object.ptr())});
}

template <typename T>
py::object nativeToPython(const std::shared_ptr<T>& value) {
  if (!value) return py::none();
  if (auto* owner = std::get_deleter<PythonOwner<T>>(value))
    return py::reinterpret_borrow<py::object>(owner->owner->obj);
  auto result = py::cast(value.get(), py::return_value_policy::reference);
  // Tensor/Buffer use unique pybind holders. Keep a native shared owner alive
  // without asking pybind to change their existing holder type.
  py::capsule owner(new std::shared_ptr<T>(value), [](void* ptr) {
    delete static_cast<std::shared_ptr<T>*>(ptr);
  });
  py::detail::keep_alive_impl(result, owner);
  return result;
}

base::Any fromPython(py::handle value) {
  if (py::isinstance<py::bool_>(value)) return base::Any(value.cast<bool>());
  if (py::isinstance<py::int_>(value)) {
    auto number = value.cast<int64_t>();
    if (number >= std::numeric_limits<int>::min() && number <= std::numeric_limits<int>::max())
      return base::Any(static_cast<int>(number));
    return base::Any(number);
  }
  if (py::isinstance<py::float_>(value)) return base::Any(value.cast<double>());
  if (py::isinstance<py::str>(value)) return base::Any(value.cast<std::string>());
  if (py::isinstance<device::Tensor>(value)) return base::Any(retainNative<device::Tensor>(value));
  if (py::isinstance<device::Buffer>(value)) return base::Any(retainNative<device::Buffer>(value));
  if (py::isinstance<base::Param>(value))
    return base::Any(value.cast<std::shared_ptr<base::Param>>());
  // Copying Any only copies this shared_ptr, never Python refcounts under a lock.
  return base::Any(std::make_shared<PyObjectWrapper>(value.ptr()));
}

py::object toPython(const base::Any& value) {
  if (value.type() == typeid(bool)) return py::cast(base::get<bool>(value));
  if (value.type() == typeid(int)) return py::cast(base::get<int>(value));
  if (value.type() == typeid(int64_t)) return py::cast(base::get<int64_t>(value));
  if (value.type() == typeid(double)) return py::cast(base::get<double>(value));
  if (value.type() == typeid(float)) return py::cast(base::get<float>(value));
  if (value.type() == typeid(std::string)) return py::cast(base::get<std::string>(value));
  if (value.type() == typeid(std::shared_ptr<device::Tensor>))
    return nativeToPython(base::get<std::shared_ptr<device::Tensor>>(value));
  if (value.type() == typeid(std::shared_ptr<device::Buffer>))
    return nativeToPython(base::get<std::shared_ptr<device::Buffer>>(value));
  if (value.type() == typeid(std::shared_ptr<base::Param>))
    return py::cast(base::get<std::shared_ptr<base::Param>>(value));
  if (value.type() == typeid(std::shared_ptr<PyObjectWrapper>))
    return py::reinterpret_borrow<py::object>(base::get<std::shared_ptr<PyObjectWrapper>>(value)->obj);
  throw py::type_error("State field has no registered Python conversion");
}
}  // namespace

py::object getPythonState(const Node& node) {
  auto state = node.getState();
  if (!state) throw std::runtime_error("No State is bound; use graph.run_with_state(state)");
  auto storage = py::cast(state);
  if (py::hasattr(storage, "_owner")) {
    auto owner = storage.attr("_owner")();
    if (!owner.is_none()) return owner;
  }
  return py::module_::import("nndeploy.dag.state").attr("State").attr("_from_native")(storage);
}

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<State, std::shared_ptr<State>>(m, "_StateStorage", py::dynamic_attr())
      .def(py::init<>())
      .def("set", [](State& self, const std::string& key, py::object value) {
        return self.set(key, fromPython(value));
      })
      .def("get", [](const State& self, const std::string& key) {
        base::Any value;
        if (self.get(key, value) != base::kStatusCodeOk) throw py::key_error(key);
        return toPython(value);
      })
      .def("contains", &State::contains)
      .def("erase", &State::erase);
}

}  // namespace dag
}  // namespace nndeploy
