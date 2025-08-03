
include(ExternalProject)
include(cmake/util.cmake)

set(ENABLE_NNDEPLOY_PYBIND11 OFF)
if (ENABLE_NNDEPLOY_PYTHON STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_PYTHON STREQUAL "ON")
  set(ENABLE_NNDEPLOY_PYBIND11 ON)
  # add_definitions(-DENABLE_NNDEPLOY_PYTHON)
endif()

if (ENABLE_NNDEPLOY_PYBIND11)
  # 关闭不需要的pybind11功能
  # set(PYBIND11_TEST OFF CACHE BOOL "" FORCE) 
  # set(PYBIND11_BUILD_DOC OFF CACHE BOOL "" FORCE)
  # set(PYBIND11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  # set(PYBIND11_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  # message(STATUS "ENABLE_NNDEPLOY_PYBIND11: ${ENABLE_NNDEPLOY_PYBIND11}")

  # 设置pybind11路径和库
  set(PYBIND11_ROOT ${PROJECT_SOURCE_DIR}/third_party/pybind11)
  # set(LIBS pybind11)

  # 添加pybind11子目录
  # add_subdirectory(${PYBIND11_ROOT} ${LIBS})
  add_subdirectory(${PYBIND11_ROOT})

  # 添加头文件路径
  include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

  # 添加到第三方库列表
  # find_package(Python COMPONENTS Interpreter Development REQUIRED)
  
  # 添加Python库到链接列表
  # list(APPEND NNDEPLOY_THIRD_PARTY_LIBRARY ${Python_LIBRARIES})
  # message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
  # 或者使用pybind11的方式
  # list(APPEND NNDEPLOY_THIRD_PARTY_LIBRARY pybind11::embed)
endif()