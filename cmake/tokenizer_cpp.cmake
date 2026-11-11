include(ExternalProject)

if (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP STREQUAL "ON")
  # Apply the fallible HF constructor to a build-tree copy of the pinned
  # dependency. Keep the original submodule and its nested checkouts intact.
  set(tokenizer_source "${PROJECT_SOURCE_DIR}/third_party/tokenizers-cpp")
  set(tokenizer_patched "${CMAKE_CURRENT_BINARY_DIR}/patched/tokenizers-cpp")
  set(tokenizer_patch "${CMAKE_CURRENT_LIST_DIR}/patches/tokenizers-hf-errors.patch")
  file(COPY "${tokenizer_source}/" DESTINATION "${tokenizer_patched}"
       PATTERN ".git" EXCLUDE PATTERN "target" EXCLUDE PATTERN "build" EXCLUDE)
  foreach(file CMakeLists.txt rust/src/lib.rs src/huggingface_tokenizer.cc)
    configure_file("${tokenizer_source}/${file}" "${tokenizer_patched}/${file}" COPYONLY)
  endforeach()
  find_package(Git REQUIRED)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env
                  "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/patched"
                  "${GIT_EXECUTABLE}" apply --no-index "${tokenizer_patch}"
                  WORKING_DIRECTORY "${tokenizer_patched}" RESULT_VARIABLE patch_status)
  if(NOT patch_status EQUAL 0)
    message(FATAL_ERROR "Failed to apply tokenizer error-handling patch")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${tokenizer_patch}")
  # OUTPUT commands already run sequentially; POST_BUILD is only valid for
  # the TARGET form of add_custom_command (CMP0175).
  file(READ "${tokenizer_patched}/CMakeLists.txt" tokenizer_cmake)
  string(REPLACE "POST_BUILD COMMAND" "COMMAND"
                 tokenizer_cmake "${tokenizer_cmake}")
  file(WRITE "${tokenizer_patched}/CMakeLists.txt" "${tokenizer_cmake}")
  add_subdirectory("${tokenizer_patched}" "${CMAKE_CURRENT_BINARY_DIR}/third_party/tokenizers-cpp")
  include_directories("${tokenizer_source}/include")
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} tokenizers_cpp)   

  # find_library(PROTOBUF_LITE_LIB protobuf-lite REQUIRED)
  # if (PROTOBUF_LITE_LIB)
  #   message(STATUS "Found protobuf-lite: ${PROTOBUF_LITE_LIB}")
  #   set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${PROTOBUF_LITE_LIB})   
  # else()
  #   message(STATUS "Not found protobuf-lite, try to build it")
  #   set(LIBS libprotobuf-lite)
  #   add_subdirectory_if_no_target(${PROJECT_SOURCE_DIR}/third_party/protobuf ${LIBS})
  #   include_directories(${PROBUF_ROOT}/src)
  #   set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${LIBS})
  # endif()

  # set(LIBS libprotobuf-lite)
  # add_subdirectory_if_no_target(${PROJECT_SOURCE_DIR}/third_party/protobuf ${LIBS})
  # include_directories(${PROBUF_ROOT}/src)
  # set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${LIBS})

else()
endif()
