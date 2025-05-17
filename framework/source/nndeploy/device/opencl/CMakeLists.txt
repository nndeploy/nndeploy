cmake_minimum_required(VERSION 3.10)
project(OpenCLTest)

find_package(OpenCL REQUIRED)

add_executable(test_opencl
  opencl_device.cpp     
)

# set(ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# set(
#   DEVICE_INCLUDE_DIRS 
#   ${ROOT_PATH}/framework/include
# )

target_include_directories(test_opencl
  PRIVATE 
  ${OpenCL_INCLUDE_DIRS}
  C:/Users/james/source/repos/nndeploy/framework/include
)

target_link_libraries(test_opencl
  PRIVATE
    ${OpenCL_LIBRARIES}
)

get_target_property(TEST_INC_DIRS test_opencl INCLUDE_DIRECTORIES)
message(STATUS "test_opencl include directories: ${TEST_INC_DIRS}")

