#pragma once

#include <pybind11/pybind11.h>

// Constructor arguments are (name, inputs, outputs, ...). Pin each Edge,
// since callers may clear or reuse the Python lists after construction.
struct KeepNodeEdges {};

namespace pybind11 {
namespace detail {
template <>
struct process_attribute<KeepNodeEdges> : process_attribute_default<KeepNodeEdges> {
  static void postcall(function_call &call, handle) {
    for (size_t index : {2u, 3u}) {
      for (auto edge : reinterpret_borrow<iterable>(call.args[index]))
        keep_alive_impl(call.init_self, edge);
    }
  }
};
}  // namespace detail
}  // namespace pybind11
