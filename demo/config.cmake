message(STATUS "Building nndeploy demo")

set(NNDEPLOY_DEPEND_LIBRARY_DEMO)
set(NNDEPLOY_SYSTEM_LIBRARY_DEMO)
set(NNDEPLOY_DEMO_THIRD_PARTY_LIBRARY)

if (ENABLE_NNDEPLOY_DAG)
  include(${ROOT_PATH}/demo/dag/config.cmake)
endif()

if (ENABLE_NNDEPLOY_PLUGIN_DETECT)
  include(${ROOT_PATH}/demo/detect/config.cmake)
endif()

if (ENABLE_NNDEPLOY_PLUGIN_SEGMENT)
  include(${ROOT_PATH}/demo/segment/config.cmake)
endif()

if (ENABLE_NNDEPLOY_NET)
  include(${ROOT_PATH}/demo/test_net/config.cmake)
endif()

if (ENABLE_NNDEPLOY_OP)
  include(${ROOT_PATH}/demo/test_op/config.cmake)
endif()

if (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
  include(${ROOT_PATH}/demo/tokenizer_cpp/config.cmake)
endif()
