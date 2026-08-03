# cmake/Eigen3Config.cmake
# Shim config file so boxmot's find_package(Eigen3 REQUIRED NO_MODULE) finds
# the third_party/eigen clone without installing Eigen system-wide.
#
# Eigen3_DIR is set to this directory by cmake/eigen.cmake before boxmot is
# added via add_subdirectory.

if(TARGET Eigen3::Eigen)
  return()
endif()

set(_EIGEN3_ROOT "${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen")

if(EXISTS "${_EIGEN3_ROOT}/Eigen/Core")
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_EIGEN3_ROOT}"
  )
  set(Eigen3_FOUND TRUE)
  set(EIGEN3_FOUND TRUE)
  message(STATUS "Eigen3Config shim: found Eigen headers at ${_EIGEN3_ROOT}")
else()
  message(FATAL_ERROR
    "Eigen3Config shim: Eigen/Core not found at ${_EIGEN3_ROOT}\n"
    "Run: cd third_party && git clone --depth 1 --branch 3.4.0 "
    "https://gitlab.com/libeigen/eigen.git eigen")
endif()
