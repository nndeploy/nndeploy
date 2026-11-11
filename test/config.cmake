enable_testing()

function(add_nndeploy_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME}
    ${TEST_SOURCE}
  )
  target_link_libraries(${TEST_NAME}
    GTest::gtest_main
  )
  if (APPLE)
    set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
  elseif (UNIX)
    set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
  elseif(WIN32)
    if(MSVC)
      # target_link_options(${TEST_NAME} PRIVATE /WHOLEARCHIVE)
    elseif(MINGW)
      set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
    endif()
  endif()
  # DEPEND_LIBRARY
  target_link_libraries(${TEST_NAME} ${DEPEND_LIBRARY}) 
  # SYSTEM_LIBRARY
  target_link_libraries(${TEST_NAME} ${SYSTEM_LIBRARY}) 
  # THIRD_PARTY_LIBRARY
  target_link_libraries(${TEST_NAME} ${THIRD_PARTY_LIBRARY}) 
  # install
  if(SYSTEM_Windows)
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_TEST_PATH})
  else() 
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_TEST_PATH})
  endif()

  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

set(DIRECTORY test)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_FRAMEWORK_BINARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEPEND_LIBRARY})
# SYSTEM_LIBRARY
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_SYSTEM_LIBRARY})
# THIRD_PARTY_LIBRARY
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_LIST})

set(TEST_BUILD_DIR ${CMAKE_BINARY_DIR}/test)

add_nndeploy_test(edge_test ${ROOT_PATH}/test/dag/edge_test.cc)
add_nndeploy_test(dag_regression_test ${ROOT_PATH}/test/dag/dag_regression_test.cc)
set_tests_properties(dag_regression_test PROPERTIES TIMEOUT 30)
add_nndeploy_test(graph_test ${ROOT_PATH}/test/dag/graph_test.cc)
add_nndeploy_test(executor_regression_test ${ROOT_PATH}/test/dag/executor_regression_test.cc)
set_tests_properties(executor_regression_test PROPERTIES TIMEOUT 30)
add_nndeploy_test(device_buffer_tensor_test
                  ${ROOT_PATH}/test/device/device_buffer_tensor_test.cc)

add_nndeploy_test(scalar_output_test ${ROOT_PATH}/test/op/scalar_output_test.cc)

add_nndeploy_test(thread_pool_concurrency_test ${ROOT_PATH}/test/thread_pool/concurrency_test.cc)
set_tests_properties(thread_pool_concurrency_test PROPERTIES TIMEOUT 60)

if(ENABLE_NNDEPLOY_INFERENCE_DEFAULT)
  if(TARGET nndeploy_plugin_infer)
    add_nndeploy_test(infer_input_test ${ROOT_PATH}/test/infer/infer_input_test.cc)
    target_link_libraries(infer_input_test nndeploy_plugin_infer)
  endif()
  add_nndeploy_test(inference_lifecycle_test ${ROOT_PATH}/test/inference/inference_lifecycle_test.cc)
endif()

if(TARGET nndeploy_plugin_preprocess AND ENABLE_NNDEPLOY_OPENCV)
  add_nndeploy_test(batch_preprocess_test ${ROOT_PATH}/test/preprocess/batch_preprocess_test.cc)
  target_link_libraries(batch_preprocess_test nndeploy_plugin_preprocess)
endif()

if(TARGET nndeploy_plugin_llm)
  add_nndeploy_test(sampler_regression_test ${ROOT_PATH}/test/plugins/sampler_regression_test.cc)
  target_link_libraries(sampler_regression_test nndeploy_plugin_llm)
endif()
if(TARGET nndeploy_plugin_segment)
  add_nndeploy_test(segmask_lifetime_test ${ROOT_PATH}/test/plugins/segmask_lifetime_test.cc)
endif()

foreach(plugin qwen stable_diffusion super_resolution depth)
  if(TARGET nndeploy_plugin_${plugin})
    add_nndeploy_test(${plugin}_regression_test
                      ${ROOT_PATH}/test/plugins/${plugin}_regression_test.cc)
    target_link_libraries(${plugin}_regression_test nndeploy_plugin_${plugin})
  endif()
endforeach()

unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
