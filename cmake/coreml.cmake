include(ExternalProject)

if (ENABLE_NNDEPLOY_INFERENCE_COREML STREQUAL "OFF")
else()
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} "/System/Library/Frameworks/CoreML.framework")
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} "/System/Library/Frameworks/CoreVideo.framework")
endif()