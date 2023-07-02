# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_device)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/nndeploy/source/device/*.h"
  "${ROOT_PATH}/nndeploy/source/device/*.cc"
  )

if (NNDEPLOY_ENABLE_DEVICE_CPU)
  include(cpu/config.cmake)
endif()
if (NNDEPLOY_ENABLE_DEVICE_X86)
  include(x86/config.cmake)
endif()
if (NNDEPLOY_ENABLE_DEVICE_ARM)
  include(arm/config.cmake)
endif()
if (NNDEPLOY_ENABLE_DEVICE_CUDA)
  include(cuda/config.cmake)
endif()
if (NNDEPLOY_ENABLE_DEVICE_OPENCL)
  include(opencl/config.cmake)
endif()
if (NNDEPLOY_ENABLE_DEVICE_METAL)
  include(metal/config.cmake)
endif()

# OBJECT

# TARGET
add_library(${BINARY} ${NNDEPLOY_LIB_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY nndeploy_base)
target_link_libraries(${BINARY} ${DEPEND_LIBRARY})

# SYSTEM_LIBRARY
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY})

# THIRD_PARTY_LIBRARY
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}) 

# install
install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)