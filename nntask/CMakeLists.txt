# demo
option(NNKIT_TASK_ENABLE_DEMO "" ON)

# leetcode
option(NNKIT_TASK_ENABLE_LEETCODE "" OFF)
if(NNKIT_TASK_ENABLE_LEETCODE)
  add_subdirectory(source/leetcode)
endif()

# xuanxuan
option(NNKIT_TASK_ENABLE_XUANXUAN "" ON)
if(NNKIT_TASK_ENABLE_XUANXUAN)
  add_subdirectory(source/xuanxuan)
endif()
