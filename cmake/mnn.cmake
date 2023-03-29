
include(ExternalProject)

if (NNDEPLOY_ENABLE_INFERENCE_MNN STREQUAL "ON")
  message(STATUS "NNDEPLOY_ENABLE_INFERENCE_MNN: ${NNDEPLOY_ENABLE_INFERENCE_MNN}")
else()
  message(STATUS "NNDEPLOY_ENABLE_INFERENCE_MNN: ${NNDEPLOY_ENABLE_INFERENCE_MNN}")
  include_directories(${NNDEPLOY_ENABLE_INFERENCE_MNN}/include)
  message(STATUS "include_directories(${NNDEPLOY_ENABLE_INFERENCE_MNN})")
  set(tmp_name "MNN")
  set(tmp_path ${NNDEPLOY_ENABLE_INFERENCE_MNN}/lib/x86/Release)
  set(full_name ${tmp_path}/${NNDEPLOY_LIB_PREFIX}${tmp_name}${NNDEPLOY_LIB_SUFFIX})
  set(TMP_THIRD_PARTY_LIBRARY ${TMP_THIRD_PARTY_LIBRARY} ${full_name})
endif()