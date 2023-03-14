
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
  string* name_start = &name;
}

void end(const std::string& name) {
  clock_t end;
  string* name_end = name;
  if (*name_start == *start_end) {
    TimeMeasurement.duration
  }
}

}  // namespace base
}  // namespace nndeploy
