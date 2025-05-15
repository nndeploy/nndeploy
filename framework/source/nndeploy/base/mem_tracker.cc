#include "nndeploy/base/mem_tracker.h"

#ifdef ENABLE_NNDEPLOY_GPU_MEM_TRACKER

namespace nndeploy {
namespace base {

void GpuMemoryTracker::start() {
  keep_running_ = true;
  monitor_thread_ = std::thread([this]() {
    while (keep_running_) {
      size_t free_mem = 0, total_mem = 0;
      cudaMemGetInfo(&free_mem, &total_mem);
      size_t used = total_mem - free_mem;
      if (used > max_used_) {
        max_used_ = used;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
  });
}

void GpuMemoryTracker::stop() {
  keep_running_ = false;
  if (monitor_thread_.joinable()) {
    monitor_thread_.join();
  }
}

GpuMemoryTracker mem_tracker;

void memTrackerStart() { mem_tracker.start(); }

void memTrackerEnd() { mem_tracker.stop(); }

void memTrackerPrint() { mem_tracker.print_max_used_mem(); }

}  // namespace base
}  // namespace nndeploy

#endif
