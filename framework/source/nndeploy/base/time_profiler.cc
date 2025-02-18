
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
      int index = (records_[key]->call_times_ - 1) % max_size_;
      records_[key]->cost_time_[index] = cost_time;
    } else {
      // NNDEPLOY_LOGE("name %s has ended\n", key.c_str());
      ;
    }
  }
}

float TimeProfiler::getCostTime(const std::string &key) const {
  if (records_.find(key) == records_.end()) {
    return -1.0f;
  }
  std::shared_ptr<Record> record = records_.find(key)->second;
  int index = (record->call_times_ - 1) % max_size_;
  return record->cost_time_[index];
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
  std::string name = "name";
  int name_size = static_cast<int>(name.size());
  std::string call_times = "call_times";
  int call_times_size = static_cast<int>(call_times.size());
  std::string sum_cost_time = "sum cost_time(ms)";
  int sum_cost_time_size = static_cast<int>(sum_cost_time.size());
  std::string avg_cost_time = "avg cost_time(ms)";
  int avg_cost_time_size = static_cast<int>(avg_cost_time.size());
  std::string gflops = "gflops";
  int gflops_size = static_cast<int>(gflops.size());
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      if (it->key_.size() > name_size) {
        name_size = static_cast<int>(it->key_.size());
      }
      if (std::to_string(it->call_times_).size() > call_times_size) {
        call_times_size =
            static_cast<int>(std::to_string(it->call_times_).size());
      }
      std::string sum_cost_time_str =
          std::to_string(static_cast<float>(it->cost_time_sum_) / 1000.0f);
      sum_cost_time_str =
          sum_cost_time_str.substr(0, sum_cost_time_str.find(".") + 4);
      if (sum_cost_time_str.size() > sum_cost_time_size) {
        sum_cost_time_size = static_cast<int>(sum_cost_time_str.size());
      }
      std::string avg_cost_time_str = std::to_string(
          static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_);
      avg_cost_time_str =
          avg_cost_time_str.substr(0, avg_cost_time_str.find(".") + 4);
      if (avg_cost_time_str.size() > avg_cost_time_size) {
        avg_cost_time_size = static_cast<int>(avg_cost_time_str.size());
      }
      std::string gflops_str = std::to_string(it->flops_);
      gflops_str = gflops_str.substr(0, gflops_str.find(".") + 4);
      if (gflops_str.size() > gflops_size) {
        gflops_size = static_cast<int>(gflops_str.size());
      }
    }
  }
  int total_len = name_size + 2 + call_times_size + 2 + sum_cost_time_size + 2 +
                  avg_cost_time_size + 2 + gflops_size + 2;
  std::string line(total_len, '-');
  printf("%s\n", line.c_str());
  printf("%-*s  %-*s  %-*s  %-*s  %-*s\n", static_cast<int>(name_size),
         name.c_str(), static_cast<int>(call_times_size), call_times.c_str(),
         static_cast<int>(sum_cost_time_size), sum_cost_time.c_str(),
         static_cast<int>(avg_cost_time_size), avg_cost_time.c_str(),
         static_cast<int>(gflops_size), gflops.c_str());
  printf("%s\n", line.c_str());
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      std::string name = it->key_;
      printf("%-*s  %-*d  %-*.3f  %-*.3f  %-*.3f\n",
             static_cast<int>(name_size), name.c_str(),
             static_cast<int>(call_times_size), it->call_times_,
             static_cast<int>(sum_cost_time_size),
             static_cast<float>(it->cost_time_sum_) / 1000.0f,
             static_cast<int>(avg_cost_time_size),
             static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_,
             static_cast<int>(gflops_size), it->flops_);
    }
  }
  printf("%s\n", line.c_str());
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
  std::string name = "name";
  int name_size = static_cast<int>(name.size());
  std::string call_times = "call_times";
  int call_times_size = static_cast<int>(call_times.size());
  std::string cost_time = "cost_time(ms)";
  int cost_time_size = static_cast<int>(cost_time.size());
  std::string avg_cost_time = "avg cost_time(ms)";
  int avg_cost_time_size = static_cast<int>(avg_cost_time.size());
  std::string gflops = "gflops";
  int gflops_size = static_cast<int>(gflops.size());
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      if (it->key_.size() > name_size) {
        name_size = static_cast<int>(it->key_.size());
      }
      if (std::to_string(it->call_times_).size() > call_times_size) {
        call_times_size =
            static_cast<int>(std::to_string(it->call_times_).size());
      }
      std::string cost_time_str =
          std::to_string(static_cast<float>(it->cost_time_[index]) / 1000.0f);
      cost_time_str = cost_time_str.substr(0, cost_time_str.find(".") + 4);
      if (cost_time_str.size() > cost_time_size) {
        cost_time_size = static_cast<int>(cost_time_str.size());
      }
      std::string avg_cost_time_str = std::to_string(
          static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_);
      avg_cost_time_str =
          avg_cost_time_str.substr(0, avg_cost_time_str.find(".") + 4);
      if (avg_cost_time_str.size() > avg_cost_time_size) {
        avg_cost_time_size = static_cast<int>(avg_cost_time_str.size());
      }
      std::string gflops_str = std::to_string(it->flops_);
      gflops_str = gflops_str.substr(0, gflops_str.find(".") + 4);
      if (gflops_str.size() > gflops_size) {
        gflops_size = static_cast<int>(gflops_str.size());
      }
    }
  }
  int total_len = name_size + 2 + call_times_size + 2 + cost_time_size + 2 +
                  avg_cost_time_size + 2 + gflops_size + 2;
  std::string line(total_len, '-');
  printf("%s\n", line.c_str());
  printf("%-*s  %-*s  %-*s  %-*s  %-*s\n", static_cast<int>(name_size), "name",
         static_cast<int>(call_times_size), "call_times",
         static_cast<int>(cost_time_size), "cost_time(ms)",
         static_cast<int>(avg_cost_time_size), "avg cost_time(ms)",
         static_cast<int>(gflops_size), "gflops");
  printf("%s\n", line.c_str());
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      if (index < it->call_times_) {
        std::string name = it->key_;
        index = index % max_size_;
        printf(
            "%-*s  %-*d  %-*.3f  %-*.3f  %-*.3f\n", static_cast<int>(name_size),
            name.c_str(), static_cast<int>(call_times_size), it->call_times_,
            static_cast<int>(cost_time_size),
            static_cast<float>(it->cost_time_[index]) / 1000.0f,
            static_cast<int>(avg_cost_time_size),
            static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_,
            static_cast<int>(gflops_size), it->flops_);
      }
    }
  }
  printf("%s\n", line.c_str());
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
  std::string name = "name";
  int name_size = static_cast<int>(name.size());
  std::string call_times = "call_times";
  int call_times_size = static_cast<int>(call_times.size());
  std::string sum_cost_time = "sum_cost_time(ms)";
  int sum_cost_time_size = static_cast<int>(sum_cost_time.size());
  std::string avg_cost_time = "avg cost_time(ms)";
  int avg_cost_time_size = static_cast<int>(avg_cost_time.size());
  std::string avg_cost_time_remove_warmup = "avg cost_time(ms)(remove warmup)";
  int avg_cost_time_remove_warmup_size =
      static_cast<int>(avg_cost_time_remove_warmup.size());
  std::string gflops = "gflops";
  int gflops_size = static_cast<int>(gflops.size());
  for (auto &it : records) {
    if (it->type_ == kEnd) {
      if (it->key_.size() > name_size) {
        name_size = static_cast<int>(it->key_.size());
      }
      if (std::to_string(it->call_times_).size() > call_times_size) {
        call_times_size =
            static_cast<int>(std::to_string(it->call_times_).size());
      }
      std::string sum_cost_time_str =
          std::to_string(static_cast<float>(it->cost_time_sum_) / 1000.0f);
      sum_cost_time_str =
          sum_cost_time_str.substr(0, sum_cost_time_str.find(".") + 4);
      if (sum_cost_time_str.size() > sum_cost_time_size) {
        sum_cost_time_size = static_cast<int>(sum_cost_time_str.size());
      }
      std::string avg_cost_time_str = std::to_string(
          static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_);
      avg_cost_time_str =
          avg_cost_time_str.substr(0, avg_cost_time_str.find(".") + 4);
      if (avg_cost_time_str.size() > avg_cost_time_size) {
        avg_cost_time_size = static_cast<int>(avg_cost_time_str.size());
      }
      if (avg_cost_time_size > avg_cost_time_remove_warmup_size) {
        avg_cost_time_remove_warmup_size = avg_cost_time_size;
      }
      std::string gflops_str = std::to_string(it->flops_);
      gflops_str = gflops_str.substr(0, gflops_str.find(".") + 4);
      if (gflops_str.size() > gflops_size) {
        gflops_size = static_cast<int>(gflops_str.size());
      }
    }
  }
  int total_len = name_size + 2 + call_times_size + 2 + sum_cost_time_size + 2 +
                  avg_cost_time_size + 2 + avg_cost_time_remove_warmup_size +
                  2 + gflops_size + 2;
  std::string line(total_len, '-');
  printf("%s\n", line.c_str());
  printf("%-*s  %-*s  %-*s  %-*s  %-*s  %-*s\n", static_cast<int>(name_size),
         "name", static_cast<int>(call_times_size), "call_times",
         static_cast<int>(sum_cost_time_size), "cost_time(ms)",
         static_cast<int>(avg_cost_time_size), "avg cost_time(ms)",
         static_cast<int>(avg_cost_time_remove_warmup_size),
         "avg cost_time(ms)(remove warmup)", static_cast<int>(gflops_size),
         "gflops");
  printf("%s\n", line.c_str());
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
        std::string name = it->key_;
        printf(
            "%-*s  %-*d  %-*.3f  %-*.3f  %-*.3f  %-*.3f\n",
            static_cast<int>(name_size), name.c_str(),
            static_cast<int>(call_times_size), it->call_times_,
            static_cast<int>(sum_cost_time_size),
            static_cast<float>(it->cost_time_sum_) / 1000.0f,
            static_cast<int>(avg_cost_time_size),
            static_cast<float>(it->cost_time_sum_) / 1000.0f / it->call_times_,
            static_cast<int>(avg_cost_time_remove_warmup_size),
            static_cast<float>(cost_time) / 1000.0f / valid_count,
            static_cast<int>(gflops_size), it->flops_);
      }
    }
  }
  printf("%s\n", line.c_str());
}

TimeProfiler g_time_profiler;
const int max_size_ = 1024 * 1024;

void timeProfilerReset() { g_time_profiler.reset(); }

void timePointStart(const std::string &key) { g_time_profiler.start(key); }

void timePointEnd(const std::string &key) { g_time_profiler.end(key); }

float timeProfilerGetCostTime(const std::string &key) {
  return g_time_profiler.getCostTime(key);
}

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
