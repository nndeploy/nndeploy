
#ifndef _NNDEPLOY_BASE_TIME_PROFILER_H_
#define _NNDEPLOY_BASE_TIME_PROFILER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"

namespace nndeploy {
namespace base {

class NNDEPLOY_CC_API TimeProfiler : public NonCopyable {
 public:
  TimeProfiler();
  virtual ~TimeProfiler();

  void reset();
  void start(const std::string &key);
  void end(const std::string &key);
  float getCostTime(const std::string &key) const;
  float getAverageTime(const std::string &key) const;
  void print(const std::string &title = "");
  void printIndex(const std::string &title, uint64_t index);
  void printRemoveWarmup(const std::string &title, uint64_t warmup_times);

 private:
  enum Type {
    kStart,
    kEnd,
  };
  struct Record {
    Record(const std::string &key, int64_t order, uint64_t start, int max_size)
        : key_(key),
          type_(kStart),
          order_(order),
          call_times_(1),
          cost_time_sum_(0),
          flops_(0.0f),
          start_(start) {
      cost_time_.resize(max_size);
    }
    Record(const std::string &key, int64_t order, uint64_t start, float flops,
           int max_size)
        : key_(key),
          type_(kStart),
          order_(order),
          call_times_(1),
          cost_time_sum_(0),
          flops_(flops),
          start_(start) {
      cost_time_.resize(max_size);
    }
    std::string key_;
    Type type_;
    int order_;
    int call_times_;
    std::vector<uint64_t> cost_time_;
    uint64_t cost_time_sum_;
    float flops_;
    uint64_t start_;
  };

 private:
  int64_t order_ = 0;
  std::map<std::string, std::shared_ptr<Record>> records_;
  int max_size_ = 1024;
};

extern NNDEPLOY_CC_API void timeProfilerReset();

extern NNDEPLOY_CC_API void timePointStart(const std::string &key);

extern NNDEPLOY_CC_API void timePointEnd(const std::string &key);

extern NNDEPLOY_CC_API float timeProfilerGetCostTime(const std::string &key);

extern NNDEPLOY_CC_API float timeProfilerGetAverageTime(const std::string &key);

extern NNDEPLOY_CC_API void timeProfilerPrint(const std::string &title = "");

extern NNDEPLOY_CC_API void timeProfilerPrintIndex(const std::string &title,
                                                   uint64_t index);

extern NNDEPLOY_CC_API void timeProfilerPrintRemoveWarmup(
    const std::string &title, uint64_t warmup_times);

class NNDEPLOY_CC_API Timer {
  //
  //  AutoTime.hpp
  //  MNN
  //
  //  Created by MNN on 2018/07/27.
  //  Copyright © 2018, Alibaba Group Holding Limited
  //
 public:
  Timer();
  ~Timer();
  Timer(const Timer &) = delete;
  Timer(const Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(const Timer &&) = delete;

  // reset timer
  void reset();
  // get duration (us) from init or latest reset.
  uint64_t durationInUs();

  // Get Current Time
  uint64_t current() const { return mLastResetTime; }

 protected:
  uint64_t mLastResetTime;
};

class NNDEPLOY_CC_API AutoTime : Timer {
  //
  //  AutoTime.hpp
  //  MNN
  //
  //  Created by MNN on 2018/07/27.
  //  Copyright © 2018, Alibaba Group Holding Limited
  //
 public:
  AutoTime(int line, const char *func);
  ~AutoTime();
  AutoTime(const AutoTime &) = delete;
  AutoTime(const AutoTime &&) = delete;
  AutoTime &operator=(const AutoTime &) = delete;
  AutoTime &operator=(const AutoTime &&) = delete;

 private:
  int mLine;
  char *mName;
  uint64_t mCurrentTime;
};

}  // namespace base
}  // namespace nndeploy

#ifdef ENABLE_NNDEPLOY_TIME_PROFILER
#define NNDEPLOY_TIME_PROFILER_RESET() nndeploy::base::timeProfilerReset()
#define NNDEPLOY_TIME_POINT_START(key) nndeploy::base::timePointStart(key)
#define NNDEPLOY_TIME_POINT_END(key) nndeploy::base::timePointEnd(key)
#define NNDEPLOY_TIME_PROFILER_GET_COST_TIME(key) \
  nndeploy::base::timeProfilerGetCostTime(key)
#define NNDEPLOY_TIME_PROFILER_GET_AVERAGE_TIME(key) \
  nndeploy::base::timeProfilerGetAverageTime(key)
#define NNDEPLOY_TIME_PROFILER_PRINT(title) \
  nndeploy::base::timeProfilerPrint(title)
#define NNDEPLOY_TIME_PROFILER_PRINT_INDEX(title, index) \
  nndeploy::base::timeProfilerPrintIndex(title, index)
#define NNDEPLOY_TIME_PROFILER_PRINT_REMOVE_WARMUP(title, warmup_times) \
  nndeploy::base::timeProfilerPrintRemoveWarmup(title, warmup_times)
#define NNDEPLOY_AUTO_TIME nndeploy::base::AutoTime ___t(__LINE__, __func__)
#else
#define NNDEPLOY_TIME_PROFILER_RESET()
#define NNDEPLOY_TIME_POINT_START(key)
#define NNDEPLOY_TIME_POINT_END(key)
#define NNDEPLOY_TIME_PROFILER_GET_COST_TIME(key) -1.0f
#define NNDEPLOY_TIME_PROFILER_GET_AVERAGE_TIME(key) -1.0f
#define NNDEPLOY_TIME_PROFILER_PRINT(title)
#define NNDEPLOY_TIME_PROFILER_PRINT_INDEX(title, index)
#define NNDEPLOY_TIME_PROFILER_PRINT_REMOVE_WARMUP(title, warmup_times)
#define NNDEPLOY_AUTO_TIME
#endif

#endif  // _NNDEPLOY_BASE_TIME_PROFILER_H_