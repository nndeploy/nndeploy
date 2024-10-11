# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_demo_ascend_cl)
set(DIRECTORY demo)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

# include
include_directories(${ROOT_PATH}/demo)

# SOURCE
file(GLOB_RECURSE SOURCE
  "${ROOT_PATH}/demo/ascend_cl/*.h"
  "${ROOT_PATH}/demo/ascend_cl/*.cc"
)
file(GLOB DEMO_SOURCE
  "${ROOT_PATH}/demo/*.h"
  "${ROOT_PATH}/demo/*.cc"
)
set(SOURCE ${SOURCE} ${DEMO_SOURCE})

# OBJECT
# BINARY
# add_executable(${BINARY} ${SOURCE} ${SOURCE})
# 检查是否存在 ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake 文件
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    # 如果存在，设置 ASCENDC_CMAKE_DIR 变量为该路径
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
# 否则，检查是否存在 ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake 文件
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    # 如果存在，设置 ASCENDC_CMAKE_DIR 变量为该路径
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
# 如果以上两个路径都不存在
else()
    # 输出错误信息并终止 CMake 进程
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist,please check whether the cann package is installed")
endif()

# 引入 ASCENDC_CMAKE_DIR 路径下的 ascendc.cmake 文件
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# 使用 ascendc_library 函数生成一个名为 ascendc_kernels_${RUN_MODE} 的共享库，库文件包含 KERNEL_FILES 列表中的所有文件
ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

file(GLOB KERNEL_FILES ${ROOT_PATH}/demo/ascend_cl/permute_kernel.cc)

add_executable(${BINARY} ${ROOT_PATH}/demo/ascend_cl/permute_test.cc)

set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")

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
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}
  ascendc_kernels_npu
  host_intf_pub
)

# install
if(SYSTEM.Windows)
  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_BIN_PATH})
else()
  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
endif()

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)