[1mdiff --git a/CMakeLists.txt b/CMakeLists.txt[m
[1mindex d20af08f..dfd2040e 100644[m
[1m--- a/CMakeLists.txt[m
[1m+++ b/CMakeLists.txt[m
[36m@@ -31,7 +31,8 @@[m [mnndeploy_option(ENABLE_NNDEPLOY_CXX20_ABI "Use C++20 ABI" OFF)[m
 nndeploy_option(ENABLE_NNDEPLOY_OPENMP "Enable OpenMP for parallel computing" OFF)[m
 nndeploy_option(ENABLE_NNDEPLOY_ADDRESS_SANTIZER "Enable Address Sanitizer for memory error detection" OFF)[m
 nndeploy_option(ENABLE_NNDEPLOY_DOCS "Build documentation" OFF)[m
[31m-nndeploy_option(ENABLE_NNDEPLOY_TIME_PROFILER "Enable time profiling for performance analysis" ON)[m
[32m+[m[32mnndeploy_option(ENABLE_NNDEPLOY_TIME_PROFILER "Enable time profiling for performance analysis" OFF)[m
[32m+[m[32mnndeploy_option(ENABLE_NNDEPLOY_GPU_MEM_TRACKER "Enable gpu max memory usage analysis" ON)[m
 nndeploy_option(ENABLE_NNDEPLOY_RAPIDJSON "Enable RapidJSON for JSON parsing" ON)[m
 [m
 # 2. **Core Module Options (Default Configuration Recommended)**: Controls enabling of basic modules, thread pool, device modules and other core functionalities[m
[36m@@ -297,6 +298,11 @@[m [melse()[m
   add_definitions(-DENABLE_NNDEPLOY_TIME_PROFILER)[m
 endif()[m
 [m
[32m+[m[32mif(${ENABLE_NNDEPLOY_GPU_MEM_TRACKER} MATCHES "OFF")[m
[32m+[m[32melse()[m
[32m+[m[32m  add_definitions(-DENABLE_NNDEPLOY_GPU_MEM_TRACKER)[m
[32m+[m[32mendif()[m
[32m+[m
 if(${ENABLE_NNDEPLOY_OPENCV} MATCHES "OFF")[m
 else()[m
   add_definitions(-DENABLE_NNDEPLOY_OPENCV)[m
[1mdiff --git a/demo/stable_diffusion/demo.cc b/demo/stable_diffusion/demo.cc[m
[1mindex 4fb64f1d..e14ef502 100644[m
[1m--- a/demo/stable_diffusion/demo.cc[m
[1m+++ b/demo/stable_diffusion/demo.cc[m
[36m@@ -1,5 +1,8 @@[m
[32m+[m[32m#include <cuda_runtime.h>[m
[32m+[m
 #include "flag.h"[m
 #include "nndeploy/base/glic_stl_include.h"[m
[32m+[m[32m#include "nndeploy/base/mem_tracker.h"[m
 #include "nndeploy/base/shape.h"[m
 #include "nndeploy/base/time_profiler.h"[m
 #include "nndeploy/framework.h"[m
[36m@@ -119,6 +122,7 @@[m [mint main(int argc, char* argv[]) {[m
   }[m
   NNDEPLOY_TIME_POINT_END("graph->dump()");[m
 [m
[32m+[m[32m  NNDEPLOY_MEM_TRACKER_START();[m
   NNDEPLOY_TIME_POINT_START("graph->run()");[m
   for (int i = 0; i < iter; i++) {[m
     prompt->set(prompt_text, true);[m
[36m@@ -130,6 +134,9 @@[m [mint main(int argc, char* argv[]) {[m
     }[m
   }[m
   NNDEPLOY_TIME_POINT_END("graph->run()");[m
[32m+[m[32m  NNDEPLOY_MEM_TRACKER_END();[m
[32m+[m
[32m+[m[32m  NNDEPLOY_MEM_TRACKER_PRINT();[m
 [m
   NNDEPLOY_TIME_POINT_START("graph->deinit()");[m
   status = graph->deinit();[m
