
# # safetensors
# # Disable C++ exception by default.
# option(SAFETENSORS_CPP_CXX_EXCEPTIONS "Enable C++ exception(disable by default)" OFF)
# set(SAFETENSORS_CPP_SOURCES
#   "${ROOT_PATH}/third_party/safetensors-cpp/safetensors.hh"
#   "${ROOT_PATH}/third_party/safetensors-cpp/safetensors.cc"
# )

# add_library(safetensors_cpp ${SAFETENSORS_CPP_SOURCES})

# if(NOT SAFETENSORS_CPP_CXX_EXCEPTIONS)
#   if(MSVC)
#     # TODO: disable exception reliably
#     #target_compile_options(safetensors_cpp PUBLIC /EHs-c-)
#   else()
#     target_compile_options(safetensors_cpp PUBLIC -fno-exceptions)
#   endif()
# endif()
# # target_include_directories(${NNDEPLOY_FRAMEWORK_BINARY} "${ROOT_PATH}/third_party/safetensors-cpp")
# include_directories("${ROOT_PATH}/third_party/safetensors-cpp")

include(ExternalProject)

if (ENABLE_NNDEPLOY_SAFETENSORS_CPP STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_SAFETENSORS_CPP STREQUAL "ON")
  # set(SAFETENSORS_CPP_CXX_EXCEPTIONS ON)
  option(SAFETENSORS_CPP_CXX_EXCEPTIONS "Enable C++ exception(disable by default)" ON)
  if(SYSTEM.Android)
    add_definitions(-DSAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS)
    set(SAFETENSORS_CPP_PATH ${PROJECT_SOURCE_DIR}/third_party/safetensors-cpp)
    add_subdirectory_if_no_target(${SAFETENSORS_CPP_PATH} safetensors_cpp PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/safetensors_cpp/android_madvise_fix.h)
  else()
    set(SAFETENSORS_CPP_PATH ${PROJECT_SOURCE_DIR}/third_party/safetensors-cpp)
    add_subdirectory_if_no_target(${SAFETENSORS_CPP_PATH} safetensors_cpp)
  endif()
  include_directories(${SAFETENSORS_CPP_PATH})
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} safetensors_cpp)   
else()
endif()