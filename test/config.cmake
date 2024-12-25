enable_testing() # should I call this in root cmakelists.txt
set(DAG_TEST_PATH ${ROOT_PATH}/test/source/nndeploy/dag)
add_executable(
  hello_test
  ${DAG_TEST_PATH}/hello_test.cc
)
target_link_libraries(
  hello_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(hello_test)