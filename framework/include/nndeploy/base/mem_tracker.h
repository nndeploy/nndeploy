#ifndef _NNDEPLOY_BASE_CUDA_MEMORY_TRACKER_H_
#define _NNDEPLOY_BASE_CUDA_MEMORY_TRACKER_H_

#ifdef ENABLE_NNDEPLOY_GPU_MEM_TRACKER
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"

namespace nndeploy {
namespace base {

class GpuMemoryTracker {
 public:
  GpuMemoryTracker(unsigned int interval_ms = 10)
      : keep_running_(false), max_used_(0), interval_ms_(interval_ms) {}

  void start();

  void stop();

  size_t get_max_used_bytes() const { return max_used_; }

  double get_max_used_megabytes() const {
    return max_used_ / (1024.0 * 1024.0);
  }

  void print_max_used_mem() const {
    std::cout << "Max GPU memory used: " << get_max_used_megabytes() << " MB"
              << std::endl;
  }

  ~GpuMemoryTracker() { stop(); }

 private:
  std::atomic<bool> keep_running_;
  std::thread monitor_thread_;
  size_t max_used_;
  unsigned int interval_ms_;
};

extern NNDEPLOY_CC_API void memTrackerStart();

extern NNDEPLOY_CC_API void memTrackerEnd();

extern NNDEPLOY_CC_API void memTrackerPrint();

}  // namespace base
}  // namespace nndeploy

#define NNDEPLOY_MEM_TRACKER_START() nndeploy::base::memTrackerStart()
#define NNDEPLOY_MEM_TRACKER_END() nndeploy::base::memTrackerEnd()
#define NNDEPLOY_MEM_TRACKER_PRINT() nndeploy::base::memTrackerPrint()

#else

#define NNDEPLOY_MEM_TRACKER_START()
#define NNDEPLOY_MEM_TRACKER_END()
#define NNDEPLOY_MEM_TRACKER_PRINT()

#endif

#endif  // _NNDEPLOY_BASE_CUDA_MEMORY_TRACKER_H_
