
#################### common ####################
set(GLOBAL_COMMON_THIRD_PARTY_LIBRARY)
## OpenMP
if(NNDEPLOY_ENABLE_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    add_definitions(-DNNDEPLOY_ENABLE_OPENMP)
    if(MSVC)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    else()
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})

      if(${ANDROID_NDK_MAJOR})
        if(${ANDROID_NDK_MAJOR} GREATER 20)
        else()
          link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
        endif()
      else()
        link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
      endif()
    endif()
  else()
    error("OpenMP Not Found.")
  endif()
endif()
## OpenCV

#################### common ####################


#################### base ####################
set(GLOBAL_BASE_THIRD_PARTY_LIBRARY)
#################### base ####################

#################### thread ####################
set(GLOBAL_THREAD_THIRD_PARTY_LIBRARY)
#################### thread ####################

#################### cryption ####################
set(GLOBAL_CRYPTION_THIRD_PARTY_LIBRARY)
#################### cryption ####################

#################### device ####################
set(GLOBAL_DEVICE_THIRD_PARTY_LIBRARY)
#################### device ####################

#################### op ####################
set(GLOBAL_OP_THIRD_PARTY_LIBRARY)
#################### op ####################

#################### forward ####################
set(GLOBAL_FORWARD_THIRD_PARTY_LIBRARY)
#################### forward ####################

#################### inference ####################
set(GLOBAL_INFERENCE_THIRD_PARTY_LIBRARY)
#################### inference ####################

#################### task ####################
set(GLOBAL_TASK_THIRD_PARTY_LIBRARY)
#################### task ####################

#################### test ####################
set(GLOBAL_TEST_THIRD_PARTY_LIBRARY)
#################### test ####################

#################### demo ####################
set(GLOBAL_DEMO_THIRD_PARTY_LIBRARY)
#################### demo ####################