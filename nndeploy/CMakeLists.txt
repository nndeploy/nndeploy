
if(NNDEPLOY_ENABLE_BASE)
  add_subdirectory(source/base)
endif()

if(NNDEPLOY_ENABLE_THREAD)
  add_subdirectory(source/thread)
endif()

if(NNDEPLOY_ENABLE_CRYPTION)
  add_subdirectory(source/cryption)
endif()

if(NNDEPLOY_ENABLE_DEVICE)
  add_subdirectory(source/device)
endif()

if(NNDEPLOY_ENABLE_AUDIO)
  add_subdirectory(source/audio)
endif()

if(NNDEPLOY_ENABLE_CV)
  add_subdirectory(source/cv)
endif()

if(NNDEPLOY_ENABLE_INFERENCE_DEFAULT)
  add_subdirectory(source/op)
  add_subdirectory(source/forward)
endif()

if(NNDEPLOY_ENABLE_INFERENCE)
  add_subdirectory(source/inference)
endif()

if(NNDEPLOY_ENABLE_TASK)
  add_subdirectory(source/task)
  if(NNDEPLOY_ENABLE_TASK_DETECT)
    add_subdirectory(source/task/detect)
  endif()
endif()
