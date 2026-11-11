
include(ExternalProject)
include(cmake/util.cmake)

if (ENABLE_NNDEPLOY_RAPIDJSON STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_RAPIDJSON STREQUAL "ON")
  add_definitions(-DENABLE_NNDEPLOY_RAPIDJSON)
  option(RAPIDJSON_BUILD_DOC "Build rapidjson documentation." OFF)
  option(RAPIDJSON_BUILD_EXAMPLES "Build rapidjson examples." OFF)
  option(RAPIDJSON_BUILD_TESTS "Build rapidjson perftests and unittests." OFF)
  option(RAPIDJSON_BUILD_THIRDPARTY_GTEST
    "Use gtest installation in `thirdparty/gtest` by default if available" OFF)
  set(LIBS RapidJSON)
  set(RAPIDJSON_BUILD_DOC OFF)
  set(RAPIDJSON_BUILD_EXAMPLES OFF)
  set(RAPIDJSON_BUILD_TESTS OFF)
  set(RAPIDJSON_ROOT ${PROJECT_SOURCE_DIR}/third_party/rapidjson)
  # Correct the upstream custom command in a build-tree copy so the pinned
  # submodule stays untouched and retains its install/package configuration.
  set(rapidjson_patched "${CMAKE_CURRENT_BINARY_DIR}/patched/rapidjson")
  file(COPY "${RAPIDJSON_ROOT}/" DESTINATION "${rapidjson_patched}"
       PATTERN ".git" EXCLUDE PATTERN "build" EXCLUDE)
  file(READ "${RAPIDJSON_ROOT}/CMakeLists.txt" rapidjson_cmake)
  string(REPLACE "add_custom_command(TARGET travis_doc"
                 "add_custom_command(TARGET travis_doc POST_BUILD"
                 rapidjson_cmake "${rapidjson_cmake}")
  file(WRITE "${rapidjson_patched}/CMakeLists.txt" "${rapidjson_cmake}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${RAPIDJSON_ROOT}/CMakeLists.txt")
  add_subdirectory("${rapidjson_patched}"
                   "${CMAKE_CURRENT_BINARY_DIR}/third_party/rapidjson")
  include_directories(${RAPIDJSON_ROOT}/include)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${LIBS})
else()
endif()
