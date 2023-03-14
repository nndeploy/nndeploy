
#ifndef _NNDEPLOY_INCLUDE_BASE_TIME_MEASUREMENT_H_
#define _NNDEPLOY_INCLUDE_BASE_TIME_MEASUREMENT_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"

namespace nndeploy {
namespace base {

class TimeMeasurement : public NonCopyable {
 public:
  TimeMeasurement();
  virtual ~TimeMeasurement();

  void start(const std::string &name);
  void end(const std::string &name);

  void download(const std::string &path);

 private:
  double duration;

  vector<double> time_record;

  void record(double &time_record, double duration);
};

}  // namespace base
}  // namespace nndeploy

#endif