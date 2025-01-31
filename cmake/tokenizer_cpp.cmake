include(ExternalProject)

if (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP STREQUAL "ON")
  set(TOKENZIER_CPP_PATH third_party/tokenizers-cpp)
  add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers)
  include_directories(${TOKENZIER_CPP_PATH}/include)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} tokenizers_cpp)   

  find_library(PROTOBUF_LITE_LIB protobuf-lite REQUIRED)
  if (PROTOBUF_LITE_LIB)
      message(STATUS "Found protobuf-lite: ${PROTOBUF_LITE_LIB}")
      set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${PROTOBUF_LITE_LIB})   
  else()
      message(FATAL_ERROR "protobuf-lite not found! Please install")
  endif()

else()
endif()
