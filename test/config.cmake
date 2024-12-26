enable_testing() # should I call this in root cmakelists.txt
set(DAG_TEST_PATH ${ROOT_PATH}/test/source/nndeploy/dag)

# TODO: assign the *.cc files to a variable or find a cleaner way to handle this

# edge_test
add_executable(
  edge_test
  ${ROOT_PATH}/framework/source/nndeploy/base/status.cc
  ${ROOT_PATH}/framework/source/nndeploy/base/common.cc
  ${ROOT_PATH}/framework/source/nndeploy/base/time_profiler.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/edge.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/node.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/edge/abstract_edge.cc
  ${ROOT_PATH}/framework/source/nndeploy/device/device.cc
  ${DAG_TEST_PATH}/edge_test.cc
)
target_link_libraries(
  edge_test
  GTest::gtest_main
)

# graph_test
add_executable(
  graph_test
  ${ROOT_PATH}/framework/source/nndeploy/base/status.cc
  ${ROOT_PATH}/framework/source/nndeploy/base/common.cc
  ${ROOT_PATH}/framework/source/nndeploy/base/time_profiler.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/edge.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/node.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/graph.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/util.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/executor/sequential_executor.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/executor/parallel_task_executor.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/executor/parallel_pipeline_executor.cc
  ${ROOT_PATH}/framework/source/nndeploy/dag/edge/abstract_edge.cc
  ${ROOT_PATH}/framework/source/nndeploy/device/device.cc
  ${DAG_TEST_PATH}/graph_test.cc
)

target_link_libraries(
  graph_test
  GTest::gtest_main
)

include(GoogleTest)