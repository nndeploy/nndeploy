
include(ExternalProject)
# message(STATUS "ENABLE_NNDEPLOY_INFERENCE_OPENVINO: ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}")

if (ENABLE_NNDEPLOY_INFERENCE_OPENVINO STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_INFERENCE_OPENVINO STREQUAL "ON")
else()
  include_directories(${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}/include)
  # set(OPENVINO "inference_engine"
  #              "inference_engine_legacy"
  #              "inference_engine_transformations"
  #              "inference_engine_lp_transformations"
  #              "ngraph"
  #              "tbb")
  set(OPENVINO "openvino")
  set(tmp_path ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}/lib/intel64)
  set(tmp_name ${NNDEPLOY_LIB_PREFIX}${OPENVINO}${NNDEPLOY_LIB_SUFFIX})
  set(full_name ${tmp_path}/${tmp_name})
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${full_name})
  file(GLOB_RECURSE install_libs "${tmp_path}/*")
  foreach(lib ${install_libs})
    install(FILES ${lib} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  endforeach()
endif()