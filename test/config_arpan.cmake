enable_testing() # should I call this in root CMakeLists.txt?

set(TEST_BUILD_DIR ${CMAKE_BINARY_DIR}/test)

#DAG tests
set(DAG_TEST_PATH ${ROOT_PATH}/test/source/nndeploy/dag)
set(DAG_TEST_COMMON_SOURCES
    ${ROOT_PATH}/framework/source/nndeploy/base/status.cc
    ${ROOT_PATH}/framework/source/nndeploy/base/common.cc
    ${ROOT_PATH}/framework/source/nndeploy/base/time_profiler.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/edge.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/node.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/edge/abstract_edge.cc
    ${ROOT_PATH}/framework/source/nndeploy/device/device.cc
)
set(DAG_GRAPH_TEST_SOURCES
    ${ROOT_PATH}/framework/source/nndeploy/dag/graph.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/util.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/executor/sequential_executor.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/executor/parallel_task_executor.cc
    ${ROOT_PATH}/framework/source/nndeploy/dag/executor/parallel_pipeline_executor.cc
)

function(add_test TEST_NAME TEST_SOURCES)
  if(TEST_NAME MATCHES "^dag")
    add_executable(${TEST_NAME}
      ${DAG_TEST_COMMON_SOURCES}
      ${TEST_SOURCES}
      ${DAG_TEST_PATH}/${TEST_NAME}.cc
    )
  elseif(TEST_NAME MATCHES "^op")
  endif()
  target_link_libraries(${TEST_NAME}
    GTest::gtest_main
  )
  set_target_properties(${TEST_NAME}
    PROPERTIES
    LINK_FLAGS "-Wl,--no-as-needed"
    RUNTIME_OUTPUT_DIRECTORY ${TEST_BUILD_DIR}
    )
endfunction()

add_test(dag_edge_test "")
add_test(dag_graph_test "${DAG_GRAPH_TEST_SOURCES}")

include(GoogleTest)