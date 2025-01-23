enable_testing()

function(add_test TEST_NAME TEST_SOURCES)
  add_executable(${TEST_NAME}
      ${DAG_TEST_PATH}/${TEST_NAME}.cc
  )
  target_link_libraries(${TEST_NAME}
    GTest::gtest_main
  )
  if (APPLE)
    set_target_properties(${TEST_NAME}
      PROPERTIES
      LINK_FLAGS "-Wl"
      RUNTIME_OUTPUT_DIRECTORY ${TEST_BUILD_DIR}
    )
  else ()
    set_target_properties(${TEST_NAME}
      PROPERTIES
      LINK_FLAGS "-Wl,--no-as-needed"
      RUNTIME_OUTPUT_DIRECTORY ${TEST_BUILD_DIR}
    )
  endif ()
  # DEPEND_LIBRARY
  target_link_libraries(${TEST_NAME} ${DEPEND_LIBRARY}) 
  # SYSTEM_LIBRARY
  target_link_libraries(${TEST_NAME} ${SYSTEM_LIBRARY}) 
  # THIRD_PARTY_LIBRARY
  target_link_libraries(${TEST_NAME} ${THIRD_PARTY_LIBRARY}) 
  # install
  if(SYSTEM.Windows)
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_BIN_PATH})
  else() 
    install(TARGETS ${TEST_NAME} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  endif()
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
#DAG tests
set(DAG_TEST_PATH ${ROOT_PATH}/test/source/nndeploy/dag)

add_test(edge_test "")
add_test(graph_test "")

include(GoogleTest)

unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)