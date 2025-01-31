
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
  // for (auto &it : records_) {
  //   delete it.second;
  // }
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
    Record *ptr = new Record(key, order_, start, max_size_);
    std::shared_ptr<Record> record;
    record.reset(ptr);
    records_[key] = record;
  } else {
    if (records_[key]->type_ == kEnd) {
      records_[key]->type_ = kStart;
      records_[key]->call_times_++;
      records_[key]->start_ = start;
    } else {
      // NNDEPLOY_LOGE("name %s has started\n", key.c_str());
      ;
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
      uint64_t cost_time = end - records_[key]->start_;
      records_[key]->cost_time_sum_ += cost_time;
      int index = records_[key]->call_times_ % max_size_;
      records_[key]->cost_time_[index] = cost_time;
    } else {
      // NNDEPLOY_LOGE("name %s has ended\n", key.c_str());
      ;
    }
  }
}

void TimeProfiler::print(const std::string &title) {
  std::vector<std::shared_ptr<Record>> records;
  for (auto &it : records_) {
    records.emplace_back(it.second);
  }
  std::sort(
      records.begin(), records.end(),
      [](const std::shared_ptr<Record> a, const std::shared_ptr<Record> b) {
        return a->order_ < b->order_;
      });
  printf("TimeProfiler: %s\n", title.c_str());
  printf(
      "------------------------------------------------------------------------"
      "------------------------\n");
  printf("%-30s%-20s%-20s%-20s%-20s\n", "name", "call_times",
         "sum cost_time(ms)", "avg cost_time(ms)", "gflops");
  printf(
      "------------------------------------------------------------------------"
      "------------------------\n");
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      printf("%-30s%-20d%-20.3f%-20.3f%-20.3f\n", it->key_.c_str(),
             it->call_times_, static_cast<float>(it->cost_time_sum_) / 1000.0f,
             static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_,
             it->flops_);
    }
  }
  printf(
      "------------------------------------------------------------------------"
      "------------------------\n");
}

void TimeProfiler::printIndex(const std::string &title, uint64_t index) {
  std::vector<std::shared_ptr<Record>> records;
  for (auto &it : records_) {
    records.emplace_back(it.second);
  }
  std::sort(
      records.begin(), records.end(),
      [](const std::shared_ptr<Record> a, const std::shared_ptr<Record> b) {
        return a->order_ < b->order_;
      });
  printf("TimeProfiler: %s [index: %ld]\n", title.c_str(), index);
  printf(
      "------------------------------------------------------------------------"
      "------------------------\n");
  printf("%-30s%-20s%-20s%-20s%-20s\n", "name", "call_times", "cost_time(ms)",
         "avg cost_time(ms)", "gflops");
  printf(
      "------------------------------------------------------------------------"
      "------------------------\n");
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      if (index < it->call_times_) {
        index = index % max_size_;
        printf(
            "%-30s%-20d%-20.3f%-20.3f%-20.3f\n", it->key_.c_str(),
            it->call_times_,
            static_cast<float>(it->cost_time_[index]) / 1000.0f,
            static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_,
            it->flops_);
      }
    }
  }
  printf(
      "------------------------------------------------------------------------"
      "------------------------\n");
}

void TimeProfiler::printRemoveWarmup(const std::string &title,
                                     uint64_t warmup_times) {
  std::vector<std::shared_ptr<Record>> records;
  for (auto &it : records_) {
    records.emplace_back(it.second);
  }
  std::sort(
      records.begin(), records.end(),
      [](const std::shared_ptr<Record> a, const std::shared_ptr<Record> b) {
        return a->order_ < b->order_;
      });
  printf("TimeProfiler: %s, remove warmup %ld\n", title.c_str(), warmup_times);
  printf(
      "------------------------------------------------------------------------"
      "----------------------------------------------------------------------"
      "\n");
  printf("%-35s%-20s%-20s%-20s%-40s%-20s\n", "name", "call_times",
         "cost_time(ms)", "avg cost_time(ms)",
         "avg cost_time(ms)(remove warmup)", "gflops");
  printf(
      "------------------------------------------------------------------------"
      "----------------------------------------------------------------------"
      "\n");
  for (auto &it : records) {
    uint64_t cost_time = 0.0f;
    int valid_count = 0;
    if (it->call_times_ >= max_size_) {
      for (int i = warmup_times; i < max_size_; i++) {
        cost_time += it->cost_time_[i];
        valid_count++;
      }
    } else {
      for (int i = warmup_times; i < it->call_times_; i++) {
        cost_time += it->cost_time_[i];
        valid_count++;
      }
    }
    if (it->type_ == kEnd) {
      if (valid_count > 0) {
        printf(
            "%-35s%-20d%-20.3f%-20.3f%-40.3f%-20.3f\n", it->key_.c_str(),
            it->call_times_, static_cast<float>(it->cost_time_sum_) / 1000.0f,
            static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_,
            static_cast<float>(cost_time) / 1000.0f / valid_count, it->flops_);
      }
    }
  }
  printf(
      "------------------------------------------------------------------------"
      "----------------------------------------------------------------------"
      "\n");
}

TimeProfiler g_time_profiler;
const int max_size_ = 1024 * 1024;

void timeProfilerReset() { g_time_profiler.reset(); }

void timePointStart(const std::string &key) { g_time_profiler.start(key); }

void timePointEnd(const std::string &key) { g_time_profiler.end(key); }

void timeProfilerPrint(const std::string &title) {
  g_time_profiler.print(title);
}

void timeProfilerPrintIndex(const std::string &title, uint64_t index) {
  g_time_profiler.printIndex(title, index);
}

void timeProfilerPrintRemoveWarmup(const std::string &title,
                                   uint64_t warmup_times) {
  g_time_profiler.printRemoveWarmup(title, warmup_times);
}

}  // namespace base
}  // namespace nndeploy
