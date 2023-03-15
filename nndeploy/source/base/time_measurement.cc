
#include "nndeploy/include/base/time_measurement.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"


namespace nndeploy {
namespace base {

TimeMeasurement::TimeMeasurement() {}

TimeMeasurement::start(const std::string& name) {
  clock_t start;

  auto iter = time_record.find(name);
  if (iter != time_record.end()) {
    (iter->second)*.start = start;
    record_check.push((iter->second)*);
  } else {
    time_record_seperate* name_now = new time_record_seperate(name, start);
    time_record[name] = name_now;
    record_check.push(name_now);
  }
}

void end(const std::string& name) {  //这里其实不需要输入参数也可以
  clock_t end;
  double duiration_now = end - (record_check.top()) *.start;
  (record_check.top()) *.count_++;
  (record_check.top())*.duiration_sum = (record_check.top())*.duiration_sum) +duiration_now;
  record_check.pop();
}

void download(const std::string& path) {}
}  // namespace base

}  // namespace nndeploy
}  // namespace nndeploy
