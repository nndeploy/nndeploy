
#include "nndeploy/base/time_profiler.h"
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

namespace nndeploy {
namespace base {

static inline int64_t getTime() {
  uint64_t time;
#if defined(_MSC_VER)
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  uint64_t sec = now.QuadPart / freq.QuadPart;
  uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
  time = sec * 1000000 + usec;
#else
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
  return time;
}

TimeProfiler::TimeProfiler() {}

TimeProfiler::~TimeProfiler() { reset(); }

void TimeProfiler::reset() {
  for (auto &it : records_) {
    delete it.second;
  }
  records_.clear();
  order_ = 0;
}

void TimeProfiler::start(const std::string &key) {
  if (key.empty()) {
    NNDEPLOY_LOGE("name is empty!\n");
    return;
  }
  uint64_t start = getTime();
  if (records_.find(key) == records_.end()) {
    ++order_;
    Record *record = new Record(key, order_, start);
    records_[key] = record;
  } else {
    if (records_[key]->type_ == kEnd) {
      records_[key]->type_ = kStart;
      records_[key]->call_times_++;
      records_[key]->start_ = start;
    } else {
      NNDEPLOY_LOGE("name %s has started\n", key.c_str());
    }
  }
}

void TimeProfiler::end(const std::string &key) {
  if (key.empty()) {
    NNDEPLOY_LOGE("name is empty\n");
    return;
  }
  uint64_t end = getTime();
  if (records_.find(key) == records_.end()) {
    NNDEPLOY_LOGE("name %s has not started\n", key.c_str());
  } else {
    if (records_[key]->type_ == kStart) {
      records_[key]->type_ = kEnd;
      records_[key]->cost_time_ += end - records_[key]->start_;
    } else {
      NNDEPLOY_LOGE("name %s has ended\n", key.c_str());
    }
  }
}

void TimeProfiler::print(const std::string &title) {
  std::vector<Record *> records;
  for (auto &it : records_) {
    records.push_back(it.second);
  }
  std::sort(
      records.begin(), records.end(),
      [](const Record *a, const Record *b) { return a->order_ < b->order_; });
  printf("TimeProfiler: %s\n", title.c_str());
  printf(
      "------------------------------------------------------------------------"
      "-------------------\n");
  printf("%-20s%-20s%-20s%-20s%-20s\n", "name", "call_times", "cost_time",
         "cost_time/call", "flops");
  printf(
      "------------------------------------------------------------------------"
      "-------------------\n");
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      printf("%-20s%-20d%-20.3f%-20.3f%-20.3f\n", it->key_.c_str(),
             it->call_times_, (float)(it->cost_time_ / 1000.0f),
             it->cost_time_ / 1000.0f / it->call_times_, it->flops_);
    }
  }
  printf(
      "------------------------------------------------------------------------"
      "-------------------\n");
}

TimeProfiler g_time_profiler;

void timeProfilerReset() { g_time_profiler.reset(); }

void timePointStart(const std::string &key) { g_time_profiler.start(key); }

void timePointEnd(const std::string &key) { g_time_profiler.end(key); }

void timeProfilerPrint(const std::string &title) {
  g_time_profiler.print(title);
}

}  // namespace base
}  // namespace nndeploy
