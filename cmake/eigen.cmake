# cmake/eigen.cmake
# Provides Eigen3::Eigen target for nndeploy and boxmot integration.
#
# Resolution order:
#   1. third_party/eigen/ (local clone) — preferred
#   2. System-installed Eigen3 via find_package

set(EIGEN_ROOT "${PROJECT_SOURCE_DIR}/third_party/eigen")

if(EXISTS "${EIGEN_ROOT}/Eigen/Core")
  set(EIGEN_FOUND TRUE)
  set(EIGEN_INCLUDE_DIRS "${EIGEN_ROOT}")
  message(STATUS "Found Eigen3 (local): ${EIGEN_INCLUDE_DIRS}")

  # Create Eigen3::Eigen target if not already defined.
  # Also set Eigen3_DIR so boxmot's find_package(Eigen3 REQUIRED NO_MODULE)
  # picks up our shim in cmake/Eigen3Config.cmake.
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_ROOT}"
    )
    set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE PATH "" FORCE)
  endif()
else()
  # Fallback: system-installed Eigen3
  find_package(Eigen3 QUIET NO_MODULE)
  if(Eigen3_FOUND)
    set(EIGEN_FOUND TRUE)
    message(STATUS "Found Eigen3 (system): ${EIGEN3_VERSION}")
  else()
    message(FATAL_ERROR
      "Eigen3 not found.\n"
      "  Option 1: cd third_party && git clone --depth 1 --branch 3.4.0 "
      "https://gitlab.com/libeigen/eigen.git eigen\n"
      "  Option 2: Install Eigen3 system-wide (e.g. apt install libeigen3-dev)")
  endif()
endif()
