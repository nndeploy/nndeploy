
#ifndef _NNDEPLOY_INCLUDE_BASE_TIME_MEASUREMENT_H_
#define _NNDEPLOY_INCLUDE_BASE_TIME_MEASUREMENT_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include<stack>

namespace nndeploy {
namespace base {

struct time_record_seperate {
  std::string name_;
  clock_t start_;
  double duration_sum_;
  int count_;
  time_record_seperate (std::string name, clock_t start, double dur = 0, int count = 0)
      : name_(name), start_(start), duration_sum_(dur), count_(count) {}
};

class TimeMeasurement : public NonCopyable {
 public:
  TimeMeasurement();
  virtual ~TimeMeasurement();

  void start(const std::string &name);
  void end(const std::string &name);

  void download(const std::string &path);

 private:
  std::map<std::string, time_record_seperate *> time_record;
  std::stack<time_record_seperate *> record_check;  //一个顺序的接口没有写
};

}  // namespace base
}  // namespace nndeploy

#endif