
option(NNTASK_ENABLE_SEGMENT "" OFF)
if(NNTASK_ENABLE_SEGMENT)
  add_subdirectory(source/segment)
endif()

