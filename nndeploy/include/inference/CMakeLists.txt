# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_inference)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/nndeploy/source/inference/*.h"
  "${ROOT_PATH}/nndeploy/source/inference/*.cc"
  )

# OBJECT

# include
if (NNDEPLOY_ENABLE_INFERENCE_DEFAULT)
  include(default/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_TENSORRT)
  include(tensor_rt/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_OPENVINO)
  include(openvino/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_COREML)
  include(coreml/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_TFLITE)
  include(tflite/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_ONNXRUNTIME)
  include(onnxruntime/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_NCNN)
  include(mnn/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_TNN)
  include(mnn/config.cmake)
endif()
if (NNDEPLOY_ENABLE_INFERENCE_MNN)
  include(mnn/config.cmake)
endif()

# TARGET
add_library(${BINARY} ${NNDEPLOY_LIB_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY nndeploy_base)
list(APPEND DEPEND_LIBRARY nndeploy_device)
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
