
include(ExternalProject)
include(cmake/util.cmake)

if (ENABLE_NNDEPLOY_RAPIDJSON STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_RAPIDJSON STREQUAL "ON")
  add_definitions(-DENABLE_NNDEPLOY_RAPIDJSON)
  set(LIBS RapidJSON)
  set(RAPIDJSON_ROOT ${PROJECT_SOURCE_DIR}/third_party/rapidjson)
  add_subdirectory_if_no_target(${RAPIDJSON_ROOT} ${LIBS})
  include_directories(${RAPIDJSON_ROOT}/include)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${LIBS})
else()
endif()