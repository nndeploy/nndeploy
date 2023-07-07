
#ifndef _NNDEPLOY_BASE_TIME_MEASUREMENT_H_
#define _NNDEPLOY_BASE_TIME_MEASUREMENT_H_

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

  void print(const std::string &title);

 private:
  enum Type {
    kStart,
    kEnd,
  };
  struct Record {
    Record(const std::string &key, int64_t order, uint64_t start,
           float flops = 0.0f)
        : key_(key),
          type_(kStart),
          order_(order),
          call_times_(1),
          cost_time_((uint64_t)0),
          flops_(flops),
          start_(start) {}
    std::string key_;
    Type type_;
    int order_;
    int call_times_;
    uint64_t cost_time_;
    float flops_;
    uint64_t start_;
  };

 private:
  int64_t order_ = 0;
  std::map<std::string, Record *> records_;
};

}  // namespace base
}  // namespace nndeploy

#endif