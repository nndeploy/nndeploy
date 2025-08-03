# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_demo_ascendc_op)
set(DIRECTORY demo)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

# include
include_directories(${ROOT_PATH}/demo)
include_directories(${ROOT_PATH}/framework/source/nndeploy/op/ascend_cl/ascend_c)

# SOURCE
# file(GLOB_RECURSE SOURCE
#   "${ROOT_PATH}/demo/acl_op/*.h"
#   "${ROOT_PATH}/demo/acl_op/*.cc"
# )
set(SOURCE ${ROOT_PATH}/demo/ascendc_dev/demo.cc)
# file(GLOB DEMO_SOURCE
#   "${ROOT_PATH}/demo/*.h"
#   "${ROOT_PATH}/demo/*.cc"
# )
# set(SOURCE ${SOURCE} ${DEMO_SOURCE})

# ascend c
# set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
# set(RUN_MODE "npu" CACHE STRING "run mode: npu")
# set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
# # set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_CANN_PACKAGE_PATH})
# if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
#   set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
# elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
#   set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
# elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
#   set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
# else()
#   message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
# endif()
# # include_directories(/usr/local/Ascend/ascend-toolkit/8.0.RC3/tools/tikcpp/tikcfw)
# # include_directories(/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux/ascendc/include/basic_api/impl)
# # include_directories(/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux/ascendc/include/basic_api/interface/)
# # set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/8.0.RC3/aarch64-linux/tikcpp/ascendc_kernel_cmake" CACHE PATH "ASCEND CANN package installation directory")
# include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
# message("ASCEND_CANN_PACKAGE_PATH: " ${ASCEND_CANN_PACKAGE_PATH})
# message("ASCENDC_CMAKE_DIR: " ${ASCENDC_CMAKE_DIR})

# OBJECT
# BINARY
# file(GLOB OP_SOURCE
#   "${ROOT_PATH}/demo/acl_op/*.cpp"
# )
set(OP_SOURCE ${ROOT_PATH}/framework/source/nndeploy/op/ascend_cl/ascend_c/op_add_kernel.cc)
# message("OP_SOURCE: " ${OP_SOURCE})
ascendc_library(kernels STATIC
    ${OP_SOURCE}
)
unset(OP_SOURCE)
add_executable(${BINARY} ${SOURCE} ${OBJECT})
target_link_libraries(${BINARY} kernels)
if (APPLE)
  set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif (UNIX)
  set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed -Wl,--no-undefined")
elseif(WIN32)
  if(MSVC)
    # target_link_options(${BINARY} PRIVATE /WHOLEARCHIVE)
  elseif(MINGW)
    set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_FRAMEWORK_BINARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEPEND_LIBRARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEMO_DEPEND_LIBRARY})
target_link_libraries(${BINARY} ${DEPEND_LIBRARY})

# SYSTEM_LIBRARY
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_SYSTEM_LIBRARY})
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_DEMO_SYSTEM_LIBRARY})
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY})

# THIRD_PARTY_LIBRARY
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_DEMO_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_LIST})
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY})

# install
if(SYSTEM.Windows)
  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_DEMO_PATH})
else()
  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_DEMO_PATH})
endif()

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)