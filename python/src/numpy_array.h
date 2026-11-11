#pragma once

#include <pybind11/numpy.h>

namespace nndeploy {

// Native consumers can write through the pointer and assume packed, aligned
// elements. NumPy returns the original array when it meets these requirements.
inline pybind11::array writableContiguousArray(const pybind11::array &array) {
  namespace py = pybind11;
  auto result = py::array::ensure(array, py::array::c_style |
      py::detail::npy_api::NPY_ARRAY_ALIGNED_ |
      py::detail::npy_api::NPY_ARRAY_WRITEABLE_);
  if (!result) throw py::value_error("cannot obtain contiguous writable storage");
  return result;
}

}  // namespace nndeploy
